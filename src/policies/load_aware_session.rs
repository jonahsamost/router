//! Load-aware initial placement with sticky session affinity.
//!
//! A session's first request is placed on a least-loaded healthy worker. The
//! binding is then reused for later turns so the conversation remains on the
//! worker that owns its KV cache. Load is deliberately ignored after initial
//! placement; an unavailable worker is the only reason to reassign a live
//! binding.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tracing::{debug, info};

use super::hash_key;
use super::{get_healthy_worker_indices, ConsistentHashPolicy, LoadBalancingPolicy};
use super::{RequestHeaders, RoutingInputKind};
use crate::core::{Worker, WorkerType};
use crate::metrics::RouterMetrics;

pub const DEFAULT_SESSION_BINDING_TTL_SECS: u64 = 3600;
pub const DEFAULT_MAX_SESSION_BINDINGS: usize = 100_000;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct BindingKey {
    model_id: String,
    worker_type: WorkerType,
    session_key: String,
}

#[derive(Debug, Clone)]
struct SessionBinding {
    worker_url: String,
    last_seen: Instant,
}

/// Places a new session using current worker load, then keeps it sticky.
#[derive(Debug)]
pub struct LoadAwareSessionPolicy {
    bindings: Mutex<HashMap<BindingKey, SessionBinding>>,
    binding_ttl: Duration,
    max_bindings: usize,
}

impl LoadAwareSessionPolicy {
    pub fn new() -> Self {
        Self::with_limits(
            Duration::from_secs(DEFAULT_SESSION_BINDING_TTL_SECS),
            DEFAULT_MAX_SESSION_BINDINGS,
        )
    }

    pub fn with_limits(binding_ttl: Duration, max_bindings: usize) -> Self {
        assert!(max_bindings > 0, "max_bindings must be positive");
        Self {
            bindings: Mutex::new(HashMap::new()),
            binding_ttl,
            max_bindings,
        }
    }

    fn binding_key(workers: &[Arc<dyn Worker>], session_key: String) -> BindingKey {
        BindingKey {
            model_id: workers[0].model_id().to_string(),
            worker_type: workers[0].worker_type(),
            session_key,
        }
    }

    fn find_available_worker(workers: &[Arc<dyn Worker>], worker_url: &str) -> Option<usize> {
        workers
            .iter()
            .position(|worker| worker.url() == worker_url && worker.is_available())
    }

    /// Pick the least-loaded worker. A stable rendezvous score breaks ties so
    /// sequentially created sessions do not all land on worker zero while the
    /// cluster is idle.
    fn initial_placement(
        workers: &[Arc<dyn Worker>],
        healthy_indices: &[usize],
        session_key: &str,
    ) -> usize {
        let min_load = healthy_indices
            .iter()
            .map(|&index| workers[index].load())
            .min()
            .expect("healthy worker set must not be empty");

        healthy_indices
            .iter()
            .copied()
            .filter(|&index| workers[index].load() == min_load)
            .max_by_key(|&index| {
                let score_key = format!("{}\x1f{}", session_key, workers[index].url());
                ConsistentHashPolicy::fbi_hash(&score_key)
            })
            .expect("least-loaded worker set must not be empty")
    }

    fn make_room(&self, bindings: &mut HashMap<BindingKey, SessionBinding>, now: Instant) {
        // Avoid an O(number of sessions) sweep on every first turn. Expiration
        // is lazy until the bounded table actually reaches capacity.
        if bindings.len() < self.max_bindings {
            return;
        }
        bindings.retain(|_, binding| now.duration_since(binding.last_seen) <= self.binding_ttl);
        if bindings.len() < self.max_bindings {
            return;
        }

        if let Some(oldest) = bindings
            .iter()
            .min_by_key(|(_, binding)| binding.last_seen)
            .map(|(key, _)| key.clone())
        {
            bindings.remove(&oldest);
        }
    }

    fn record_selection(&self, workers: &[Arc<dyn Worker>], index: usize) -> Option<usize> {
        let worker = &workers[index];
        worker.increment_processed();
        RouterMetrics::record_processed_request(worker.url());
        RouterMetrics::record_policy_decision(self.name(), worker.url());
        Some(index)
    }

    #[cfg(test)]
    fn binding_count(&self) -> usize {
        self.bindings.lock().unwrap().len()
    }
}

impl LoadBalancingPolicy for LoadAwareSessionPolicy {
    fn select_worker_with_headers(
        &self,
        workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        headers: Option<&RequestHeaders>,
    ) -> Option<usize> {
        let healthy_indices = get_healthy_worker_indices(workers);
        if healthy_indices.is_empty() {
            return None;
        }

        let Some(session_key) = hash_key::extract_stable_session_key(request_text, headers) else {
            // Session-less traffic remains load-aware but is deliberately not
            // retained in the sticky binding table.
            let tie_key = hash_key::extract_hash_key(request_text, headers);
            let index = Self::initial_placement(workers, &healthy_indices, &tie_key);
            return self.record_selection(workers, index);
        };
        let binding_key = Self::binding_key(workers, session_key.clone());
        let now = Instant::now();
        let mut bindings = self.bindings.lock().unwrap();

        if let Some(binding) = bindings.get_mut(&binding_key) {
            let is_fresh = now.duration_since(binding.last_seen) <= self.binding_ttl;
            if is_fresh {
                if let Some(index) = Self::find_available_worker(workers, &binding.worker_url) {
                    binding.last_seen = now;
                    let worker_url = binding.worker_url.clone();
                    drop(bindings);
                    debug!(session_key, worker_url, "reused sticky session binding");
                    return self.record_selection(workers, index);
                }
            }
            bindings.remove(&binding_key);
        }

        self.make_room(&mut bindings, now);
        let index = Self::initial_placement(workers, &healthy_indices, &session_key);
        let worker_url = workers[index].url().to_string();
        bindings.insert(
            binding_key,
            SessionBinding {
                worker_url: worker_url.clone(),
                last_seen: now,
            },
        );
        drop(bindings);

        info!(
            session_key,
            worker_url,
            load = workers[index].load(),
            "created load-aware session binding"
        );
        self.record_selection(workers, index)
    }

    fn name(&self) -> &'static str {
        "load_aware_session"
    }

    fn needs_request_text(&self) -> bool {
        true
    }

    fn routing_input_kind(&self) -> RoutingInputKind {
        RoutingInputKind::Session
    }

    fn needs_headers(&self) -> bool {
        true
    }

    fn needs_load_tracking(&self) -> bool {
        true
    }

    fn reset(&self) {
        self.bindings.lock().unwrap().clear();
        info!("Load-aware session policy reset - bindings cleared");
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Default for LoadAwareSessionPolicy {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::BasicWorker;

    fn workers() -> Vec<Arc<dyn Worker>> {
        vec![
            Arc::new(BasicWorker::new(
                "http://worker-a:8000".to_string(),
                WorkerType::Regular,
            )),
            Arc::new(BasicWorker::new(
                "http://worker-b:8000".to_string(),
                WorkerType::Regular,
            )),
        ]
    }

    #[test]
    fn new_session_chooses_least_loaded_worker_then_stays_sticky() {
        let policy = LoadAwareSessionPolicy::new();
        let workers = workers();
        workers[0].increment_load();

        let first = policy.select_worker(&workers, Some("session-a")).unwrap();
        assert_eq!(first, 1);

        workers[0].decrement_load();
        workers[1].increment_load();
        workers[1].increment_load();
        let continuation = policy.select_worker(&workers, Some("session-a")).unwrap();
        assert_eq!(continuation, first);
    }

    #[test]
    fn session_header_keeps_growing_conversation_sticky() {
        let policy = LoadAwareSessionPolicy::new();
        let workers = workers();
        workers[0].increment_load();
        let headers =
            RequestHeaders::from([("x-session-id".to_string(), "conversation-a".to_string())]);

        let first = policy
            .select_worker_with_headers(&workers, Some("first prompt"), Some(&headers))
            .unwrap();
        assert_eq!(first, 1);

        workers[0].decrement_load();
        workers[1].increment_load();
        let continuation = policy
            .select_worker_with_headers(
                &workers,
                Some("first prompt plus assistant reply and next turn"),
                Some(&headers),
            )
            .unwrap();
        assert_eq!(continuation, first);
    }

    #[test]
    fn unavailable_bound_worker_causes_reassignment() {
        let policy = LoadAwareSessionPolicy::new();
        let workers = workers();
        let first = policy.select_worker(&workers, Some("session-a")).unwrap();
        workers[first].set_healthy(false);

        let reassigned = policy.select_worker(&workers, Some("session-a")).unwrap();
        assert_ne!(reassigned, first);
        assert!(workers[reassigned].is_healthy());
    }

    #[test]
    fn equal_load_uses_session_hash_instead_of_worker_order() {
        let policy = LoadAwareSessionPolicy::new();
        let workers = workers();
        let selected: std::collections::HashSet<_> = (0..100)
            .map(|index| {
                policy
                    .select_worker(&workers, Some(&format!("session-{index}")))
                    .unwrap()
            })
            .collect();
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn reset_clears_bindings() {
        let policy = LoadAwareSessionPolicy::new();
        let workers = workers();
        policy.select_worker(&workers, Some("session-a"));
        assert_eq!(policy.binding_count(), 1);
        policy.reset();
        assert_eq!(policy.binding_count(), 0);
    }

    #[test]
    fn binding_table_is_bounded() {
        let policy = LoadAwareSessionPolicy::with_limits(Duration::from_secs(60), 2);
        let workers = workers();
        for session in ["a", "b", "c"] {
            policy.select_worker(&workers, Some(session));
        }
        assert_eq!(policy.binding_count(), 2);
    }

    #[test]
    fn request_id_does_not_create_sticky_state() {
        let policy = LoadAwareSessionPolicy::new();
        let workers = workers();
        let headers = RequestHeaders::from([(
            "x-request-id".to_string(),
            "request-only-identity".to_string(),
        )]);
        policy.select_worker_with_headers(&workers, Some(""), Some(&headers));
        assert_eq!(policy.binding_count(), 0);
    }
}
