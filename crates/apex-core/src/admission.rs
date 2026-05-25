//! Inline admission controller.
//!
//! See `docs/llds/admission-controller.md` and `docs/specs/admission-controller.md`.

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use prometheus::{Histogram, HistogramOpts, IntCounter, IntCounterVec, IntGauge, Opts, Registry};
use tokio::task::JoinHandle;

pub struct AdmissionController {
    rss: Arc<AtomicU64>,
    queue_depth: Arc<AtomicUsize>,
    max_rss_bytes: u64,
    max_queue_depth: usize,
    metrics: AdmissionMetrics,
    log_limiter: Mutex<RateLimiter>,
}

#[derive(Debug, Clone, Copy)]
pub enum AdmissionDecision {
    Admit,
    RejectMemory { rss: u64, limit: u64 },
    RejectQueueDepth { depth: usize, limit: usize },
}

impl AdmissionDecision {
    pub fn is_admit(&self) -> bool {
        matches!(self, AdmissionDecision::Admit)
    }

    pub fn reason_label(&self) -> &'static str {
        match self {
            AdmissionDecision::Admit => "admit",
            AdmissionDecision::RejectMemory { .. } => "reject_memory",
            AdmissionDecision::RejectQueueDepth { .. } => "reject_queue_depth",
        }
    }

    pub fn reject_message(&self) -> &'static str {
        match self {
            AdmissionDecision::Admit => "ok",
            AdmissionDecision::RejectMemory { .. } => "memory limit",
            AdmissionDecision::RejectQueueDepth { .. } => "queue depth limit",
        }
    }
}

#[derive(Clone)]
pub struct AdmissionMetrics {
    pub decisions: IntCounterVec,
    pub rss_bytes: IntGauge,
    pub queue_depth_gauge: IntGauge,
    pub sample_failures: IntCounter,
    pub check_duration_ns: Histogram,
}

impl AdmissionMetrics {
    pub fn register(registry: &Registry) -> Result<Self, prometheus::Error> {
        let decisions = IntCounterVec::new(
            Opts::new(
                "apex_admission_decisions_total",
                "admission decisions by outcome",
            ),
            &["outcome"],
        )?;
        let rss_bytes = IntGauge::new("apex_admission_rss_bytes", "latest sampled process RSS")?;
        let queue_depth_gauge = IntGauge::new(
            "apex_admission_queue_depth",
            "current aggregate dispatcher queue depth",
        )?;
        let sample_failures = IntCounter::new(
            "apex_admission_rss_sample_failures_total",
            "RSS sample read failures",
        )?;
        let check_duration_ns = Histogram::with_opts(
            HistogramOpts::new(
                "apex_admission_check_duration_nanoseconds",
                "admission-check hot-path latency",
            )
            .buckets(vec![10.0, 50.0, 100.0, 250.0, 500.0, 1_000.0, 10_000.0]),
        )?;
        registry.register(Box::new(decisions.clone()))?;
        registry.register(Box::new(rss_bytes.clone()))?;
        registry.register(Box::new(queue_depth_gauge.clone()))?;
        registry.register(Box::new(sample_failures.clone()))?;
        registry.register(Box::new(check_duration_ns.clone()))?;
        Ok(Self {
            decisions,
            rss_bytes,
            queue_depth_gauge,
            sample_failures,
            check_duration_ns,
        })
    }
}

impl AdmissionController {
    pub fn new(
        max_rss_bytes: u64,
        max_queue_depth: usize,
        registry: &Registry,
    ) -> Result<Self, prometheus::Error> {
        let metrics = AdmissionMetrics::register(registry)?;
        Ok(Self {
            rss: Arc::new(AtomicU64::new(0)),
            queue_depth: Arc::new(AtomicUsize::new(0)),
            max_rss_bytes,
            max_queue_depth,
            metrics,
            log_limiter: Mutex::new(RateLimiter::new(10)),
        })
    }

    /// Hot-path admission decision. Two atomic loads + integer comparisons.
    ///
    /// @spec ADMIT-CHECK-001, ADMIT-CHECK-002, ADMIT-CHECK-003,
    ///       ADMIT-CHECK-004, ADMIT-CHECK-005
    pub fn check(&self) -> AdmissionDecision {
        let rss = self.rss.load(Ordering::Relaxed);
        if rss > self.max_rss_bytes {
            return AdmissionDecision::RejectMemory {
                rss,
                limit: self.max_rss_bytes,
            };
        }
        let depth = self.queue_depth.load(Ordering::Relaxed);
        if depth > self.max_queue_depth {
            return AdmissionDecision::RejectQueueDepth {
                depth,
                limit: self.max_queue_depth,
            };
        }
        AdmissionDecision::Admit
    }

    /// Record a decision into metrics. Called after `check`.
    ///
    /// @spec ADMIT-METRIC-001, ADMIT-METRIC-004
    pub fn record(&self, decision: &AdmissionDecision, check_duration_ns: u64) {
        self.metrics
            .decisions
            .with_label_values(&[decision.reason_label()])
            .inc();
        self.metrics
            .check_duration_ns
            .observe(check_duration_ns as f64);
    }

    /// Rate-limited rejection log.
    ///
    /// @spec ADMIT-LOG-001
    pub fn maybe_log_rejection(&self, decision: &AdmissionDecision) {
        if decision.is_admit() {
            return;
        }
        let allowed = match self.log_limiter.try_lock() {
            Ok(mut limiter) => limiter.allow(),
            Err(_) => false,
        };
        if !allowed {
            return;
        }
        match decision {
            AdmissionDecision::RejectMemory { rss, limit } => {
                tracing::warn!(rss = rss, limit = limit, "admission rejected: memory");
            }
            AdmissionDecision::RejectQueueDepth { depth, limit } => {
                tracing::warn!(depth = depth, limit = limit, "admission rejected: queue depth");
            }
            AdmissionDecision::Admit => {}
        }
    }

    /// @spec ADMIT-DEPTH-001
    pub fn incr_queue(&self) {
        let depth = self.queue_depth.fetch_add(1, Ordering::Relaxed) + 1;
        self.metrics.queue_depth_gauge.set(depth as i64);
    }

    /// @spec ADMIT-DEPTH-002
    pub fn decr_queue(&self, n: usize) {
        let prev = self.queue_depth.fetch_sub(n, Ordering::Relaxed);
        debug_assert!(prev >= n, "queue depth underflow: prev={prev}, n={n}");
        self.metrics
            .queue_depth_gauge
            .set(prev.saturating_sub(n) as i64);
    }

    pub(crate) fn rss_publisher(&self) -> Arc<AtomicU64> {
        self.rss.clone()
    }

    pub(crate) fn metrics(&self) -> &AdmissionMetrics {
        &self.metrics
    }
}

/// Spawn the background RSS sampling task.
///
/// @spec ADMIT-RSS-001, ADMIT-RSS-004, ADMIT-RSS-005
pub fn spawn_rss_sampler(
    controller: Arc<AdmissionController>,
    interval: Duration,
) -> JoinHandle<()> {
    let rss = controller.rss_publisher();
    let gauge = controller.metrics().rss_bytes.clone();
    let failures = controller.metrics().sample_failures.clone();
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(interval);
        loop {
            tick.tick().await;
            match read_process_rss() {
                Ok(bytes) => {
                    rss.store(bytes, Ordering::Relaxed);
                    gauge.set(bytes as i64);
                }
                Err(e) => {
                    tracing::warn!(error = %e, "RSS sample failed");
                    failures.inc();
                }
            }
        }
    })
}

/// Read process RSS in bytes from the OS.
///
/// @spec ADMIT-RSS-002, ADMIT-RSS-003
#[cfg(target_os = "linux")]
pub(crate) fn read_process_rss() -> Result<u64, std::io::Error> {
    use std::io::Read;
    let mut s = String::new();
    std::fs::File::open("/proc/self/statm")?.read_to_string(&mut s)?;
    // statm fields (in pages): size resident shared text lib data dt
    let resident_pages: u64 = s
        .split_whitespace()
        .nth(1)
        .and_then(|p| p.parse().ok())
        .ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "malformed /proc/self/statm")
        })?;
    // SAFETY: sysconf with _SC_PAGESIZE is a documented, signal-safe call returning page size in bytes.
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64;
    Ok(resident_pages * page_size)
}

#[cfg(target_os = "macos")]
pub(crate) fn read_process_rss() -> Result<u64, std::io::Error> {
    use mach2::{
        kern_return::KERN_SUCCESS, message::mach_msg_type_number_t, task::task_info,
        task_info::MACH_TASK_BASIC_INFO, time_value::time_value_t, traps::mach_task_self,
    };

    // The mach2 crate doesn't expose `mach_task_basic_info`; define the C-compatible
    // layout here. Field order and types must match <mach/task_info.h>.
    #[repr(C)]
    #[derive(Default)]
    struct MachTaskBasicInfo {
        virtual_size: u64,
        resident_size: u64,
        resident_size_max: u64,
        user_time: time_value_t,
        system_time: time_value_t,
        policy: libc::c_int,
        suspend_count: libc::c_int,
    }

    let count = (core::mem::size_of::<MachTaskBasicInfo>() / core::mem::size_of::<u32>())
        as mach_msg_type_number_t;
    let mut info = MachTaskBasicInfo::default();
    let mut count_mut = count;
    // SAFETY: task_info is the documented Mach API for retrieving task metrics.
    // `mach_task_self` returns a port to the calling task. `info` points at properly-sized
    // stack storage with C-compatible layout matching `count_mut` u32 words.
    let kr = unsafe {
        task_info(
            mach_task_self(),
            MACH_TASK_BASIC_INFO,
            &mut info as *mut _ as *mut _,
            &mut count_mut,
        )
    };
    if kr != KERN_SUCCESS {
        return Err(std::io::Error::other(format!("task_info failed: {kr}")));
    }
    Ok(info.resident_size)
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub(crate) fn read_process_rss() -> Result<u64, std::io::Error> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "RSS sampling not implemented for this platform",
    ))
}

/// Token-bucket rate limiter. Single-thread access (caller holds mutex).
struct RateLimiter {
    capacity: u32,
    tokens: u32,
    refill_per_sec: u32,
    last_refill: Instant,
}

impl RateLimiter {
    fn new(rate_per_sec: u32) -> Self {
        Self {
            capacity: rate_per_sec,
            tokens: rate_per_sec,
            refill_per_sec: rate_per_sec,
            last_refill: Instant::now(),
        }
    }

    fn allow(&mut self) -> bool {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        let refill = (elapsed * self.refill_per_sec as f64) as u32;
        if refill > 0 {
            self.tokens = (self.tokens + refill).min(self.capacity);
            self.last_refill = now;
        }
        if self.tokens > 0 {
            self.tokens -= 1;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctrl(rss_limit: u64, depth_limit: usize) -> AdmissionController {
        AdmissionController::new(rss_limit, depth_limit, &Registry::new()).unwrap()
    }

    /// @spec ADMIT-CHECK-003
    #[test]
    fn admits_when_both_signals_under_limit() {
        let c = ctrl(1000, 10);
        c.rss.store(500, Ordering::Relaxed);
        c.queue_depth.store(5, Ordering::Relaxed);
        assert!(matches!(c.check(), AdmissionDecision::Admit));
    }

    /// @spec ADMIT-CHECK-001
    #[test]
    fn rejects_memory_when_rss_over_limit() {
        let c = ctrl(1000, 10);
        c.rss.store(1001, Ordering::Relaxed);
        match c.check() {
            AdmissionDecision::RejectMemory { rss, limit } => {
                assert_eq!(rss, 1001);
                assert_eq!(limit, 1000);
            }
            other => panic!("expected RejectMemory, got {other:?}"),
        }
    }

    /// @spec ADMIT-CHECK-002
    #[test]
    fn rejects_queue_when_depth_over_limit() {
        let c = ctrl(1000, 10);
        c.queue_depth.store(11, Ordering::Relaxed);
        match c.check() {
            AdmissionDecision::RejectQueueDepth { depth, limit } => {
                assert_eq!(depth, 11);
                assert_eq!(limit, 10);
            }
            other => panic!("expected RejectQueueDepth, got {other:?}"),
        }
    }

    #[test]
    fn memory_check_runs_before_queue_check() {
        let c = ctrl(1000, 10);
        c.rss.store(1001, Ordering::Relaxed);
        c.queue_depth.store(11, Ordering::Relaxed);
        assert!(matches!(c.check(), AdmissionDecision::RejectMemory { .. }));
    }

    /// @spec ADMIT-DEPTH-001, ADMIT-DEPTH-002
    #[test]
    fn queue_depth_counter_increments_and_decrements() {
        let c = ctrl(1000, 100);
        for _ in 0..10 {
            c.incr_queue();
        }
        assert_eq!(c.queue_depth.load(Ordering::Relaxed), 10);
        c.decr_queue(4);
        assert_eq!(c.queue_depth.load(Ordering::Relaxed), 6);
    }

    #[test]
    fn rate_limiter_grants_initial_burst_then_refuses() {
        let mut rl = RateLimiter::new(3);
        assert!(rl.allow());
        assert!(rl.allow());
        assert!(rl.allow());
        assert!(!rl.allow());
    }

    #[test]
    fn rate_limiter_refills_over_time() {
        let mut rl = RateLimiter::new(10);
        for _ in 0..10 {
            assert!(rl.allow());
        }
        assert!(!rl.allow());
        std::thread::sleep(Duration::from_millis(250));
        assert!(rl.allow(), "should have refilled after 250 ms at 10/sec");
    }

    #[test]
    fn decision_helpers() {
        assert_eq!(AdmissionDecision::Admit.reason_label(), "admit");
        assert_eq!(
            AdmissionDecision::RejectMemory { rss: 0, limit: 0 }.reason_label(),
            "reject_memory"
        );
        assert!(AdmissionDecision::Admit.is_admit());
        assert!(!AdmissionDecision::RejectMemory { rss: 0, limit: 0 }.is_admit());
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn read_process_rss_returns_nonzero_on_supported_platform() {
        let bytes = read_process_rss().expect("RSS read should succeed on supported platform");
        assert!(bytes > 0, "process should have nonzero RSS");
    }
}
