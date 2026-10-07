//! Prefix- and load-aware initial placement with sticky session affinity.
//!
//! New sessions are placed using the prompt prefix believed to be resident on
//! each worker and that worker's current active-request load. Once placed, a
//! session remains bound to its original worker so later turns reuse the
//! conversation KV cache instead of being rescored and potentially moved.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tracing::{debug, info};

use super::hash_key;
use super::{
    get_healthy_worker_indices, CacheAwareCandidate, CacheAwareConfig, CacheAwarePolicy,
    ConsistentHashPolicy, LoadBalancingPolicy, RequestHeaders, RoutingInputKind,
};
use crate::core::{Worker, WorkerType};
use crate::metrics::RouterMetrics;

use super::{DEFAULT_MAX_SESSION_BINDINGS, DEFAULT_SESSION_BINDING_TTL_SECS};

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

/// Scores only a session's initial placement, then preserves worker affinity.
#[derive(Debug)]
pub struct PrefixAwareSessionPolicy {
    affinity: CacheAwarePolicy,
    bindings: Mutex<HashMap<BindingKey, SessionBinding>>,
    binding_ttl: Duration,
    max_bindings: usize,
}

impl PrefixAwareSessionPolicy {
    const SESSION_MARKER: &'static str = "\0vllm-router-session:";

    pub fn new() -> Self {
        Self::with_config(
            Duration::from_secs(DEFAULT_SESSION_BINDING_TTL_SECS),
            DEFAULT_MAX_SESSION_BINDINGS,
            CacheAwareConfig::default(),
        )
    }

    pub fn with_config(
        binding_ttl: Duration,
        max_bindings: usize,
        cache_config: CacheAwareConfig,
    ) -> Self {
        assert!(max_bindings > 0, "max_bindings must be positive");
        Self {
            affinity: CacheAwarePolicy::with_config(cache_config),
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

    fn initial_placement(
        &self,
        workers: &[Arc<dyn Worker>],
        healthy_indices: &[usize],
        request_text: &str,
        tie_key: &str,
    ) -> Option<usize> {
        // kv_pressure = load / (1 + load) makes CacheAwarePolicy's headroom
        // exactly 1 / (1 + load). Its score therefore becomes:
        //     (1 + prefix_match_ratio) / (1 + active_requests)
        // A tiny deterministic penalty breaks exact ties without changing a
        // decision separated by a meaningful affinity or load difference.
        const TIE_EPSILON: f64 = 1e-9;
        let candidates: Vec<_> = healthy_indices
            .iter()
            .map(|&index| {
                let load = workers[index].load() as f64;
                let score_key = format!("{}\x1f{}", tie_key, workers[index].url());
                let hash_fraction =
                    ConsistentHashPolicy::fbi_hash(&score_key) as f64 / u64::MAX as f64;
                CacheAwareCandidate {
                    target_id: workers[index].url(),
                    kv_pressure: (load / (1.0 + load)) + ((1.0 - hash_fraction) * TIE_EPSILON),
                }
            })
            .collect();

        let model_id = workers[healthy_indices[0]].model_id();
        let placement =
            self.affinity
                .evaluate_initial_placement(model_id, request_text, &candidates)?;
        workers
            .iter()
            .position(|worker| worker.url() == placement.target_id)
    }

    fn make_room(&self, bindings: &mut HashMap<BindingKey, SessionBinding>, now: Instant) {
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

    fn record_selection(
        &self,
        workers: &[Arc<dyn Worker>],
        index: usize,
        request_text: &str,
    ) -> Option<usize> {
        let worker = &workers[index];
        self.affinity
            .commit_placement(worker.model_id(), request_text, worker.url());
        worker.increment_processed();
        RouterMetrics::record_processed_request(worker.url());
        RouterMetrics::record_policy_decision(self.name(), worker.url());
        Some(index)
    }

    fn split_routing_input(input: &str) -> (&str, Option<String>) {
        match input.rsplit_once(Self::SESSION_MARKER) {
            Some((prompt, session)) if !session.is_empty() => {
                (prompt, Some(format!("affinity:{session}")))
            }
            _ => (input, None),
        }
    }

    #[cfg(test)]
    fn binding_count(&self) -> usize {
        self.bindings.lock().unwrap().len()
    }
}

impl LoadBalancingPolicy for PrefixAwareSessionPolicy {
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
        let (prompt, encoded_session_key) = Self::split_routing_input(request_text.unwrap_or(""));
        let session_key =
            hash_key::extract_stable_session_key(None, headers).or(encoded_session_key);

        let Some(session_key) = session_key else {
            let tie_key = hash_key::extract_hash_key(request_text, headers);
            let index = self.initial_placement(workers, &healthy_indices, prompt, &tie_key)?;
            return self.record_selection(workers, index, prompt);
        };

        let binding_key = Self::binding_key(workers, session_key.clone());
        let now = Instant::now();
        let mut bindings = self.bindings.lock().unwrap();
        if let Some(binding) = bindings.get_mut(&binding_key) {
            if now.duration_since(binding.last_seen) <= self.binding_ttl {
                if let Some(index) = Self::find_available_worker(workers, &binding.worker_url) {
                    binding.last_seen = now;
                    let worker_url = binding.worker_url.clone();
                    drop(bindings);
                    debug!(
                        session_key,
                        worker_url, "reused prefix-aware session binding"
                    );
                    return self.record_selection(workers, index, prompt);
                }
            }
            bindings.remove(&binding_key);
        }

        self.make_room(&mut bindings, now);
        let index = self.initial_placement(workers, &healthy_indices, prompt, &session_key)?;
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
            "created prefix-aware session binding"
        );
        self.record_selection(workers, index, prompt)
    }

    fn name(&self) -> &'static str {
        "prefix_aware_session"
    }

    fn needs_request_text(&self) -> bool {
        true
    }

    fn routing_input_kind(&self) -> RoutingInputKind {
        RoutingInputKind::PrefixAndSession
    }

    fn needs_headers(&self) -> bool {
        true
    }

    fn needs_load_tracking(&self) -> bool {
        true
    }

    fn reset(&self) {
        self.bindings.lock().unwrap().clear();
        self.affinity.clear_placements();
        info!("Prefix-aware session policy reset");
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Default for PrefixAwareSessionPolicy {
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

    fn session(value: &str) -> RequestHeaders {
        RequestHeaders::from([("x-session-id".to_string(), value.to_string())])
    }

    #[test]
    fn session_stays_sticky_as_conversation_grows() {
        let policy = PrefixAwareSessionPolicy::new();
        let workers = workers();
        let headers = session("conversation-a");
        let first = policy
            .select_worker_with_headers(&workers, Some("system\nuser one"), Some(&headers))
            .unwrap();

        workers[first].increment_load();
        workers[first].increment_load();
        let continuation = policy
            .select_worker_with_headers(
                &workers,
                Some("system\nuser one\nassistant one\nuser two"),
                Some(&headers),
            )
            .unwrap();
        assert_eq!(continuation, first);
        assert_eq!(policy.binding_count(), 1);
    }

    #[test]
    fn typed_request_session_is_sticky_without_an_http_header() {
        let policy = PrefixAwareSessionPolicy::new();
        let workers = workers();
        let first = policy
            .select_worker(
                &workers,
                Some("system\nuser one\0vllm-router-session:conversation-a"),
            )
            .unwrap();
        workers[first].increment_load();
        workers[first].increment_load();
        let continuation = policy
            .select_worker(
                &workers,
                Some(
                    "system\nuser one\nassistant one\nuser two\0vllm-router-session:conversation-a",
                ),
            )
            .unwrap();
        assert_eq!(continuation, first);
    }

    #[test]
    fn shared_prefix_attracts_a_new_session_when_load_is_equal() {
        let policy = PrefixAwareSessionPolicy::new();
        let workers = workers();
        let prompt = "shared system prompt\nfirst user turn";
        let first = policy
            .select_worker_with_headers(&workers, Some(prompt), Some(&session("one")))
            .unwrap();
        let second = policy
            .select_worker_with_headers(&workers, Some(prompt), Some(&session("two")))
            .unwrap();
        assert_eq!(second, first);
    }

    #[test]
    fn sufficient_load_outweighs_a_full_prefix_match() {
        let policy = PrefixAwareSessionPolicy::new();
        let workers = workers();
        let prompt = "shared system prompt\nfirst user turn";
        let cached = policy
            .select_worker_with_headers(&workers, Some(prompt), Some(&session("one")))
            .unwrap();
        workers[cached].increment_load();
        workers[cached].increment_load();
        let placed = policy
            .select_worker_with_headers(&workers, Some(prompt), Some(&session("two")))
            .unwrap();
        assert_ne!(placed, cached);
    }
}
