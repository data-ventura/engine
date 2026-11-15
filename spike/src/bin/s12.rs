//! S12: failed reads. Injected 5xx responses, truncated chunks, timeouts and a network drop during
//! reads: affected chunks show as errors or missing, are retried, and recover. No wrong value is
//! ever drawn.
//!
//! Reads 2 m temperature steps from Google's public ARCO-ERA5 store, as S4 does, through
//! `http → FaultStore → RetryStore → Zarr → Prefetch`. The faults are injected just above the
//! network, so everything the engine does about them is real. First a clean pass records every
//! step's values. Then:
//!
//! 1. **Random faults:** 20% of attempts get a 503, 20% have their body cut in half, 5% hang, so
//!    about 4% of reads still fail after four attempts and reach the view as errors. The
//!    view steps forward every 250 ms with read-ahead, as in S4. A memory-pressure warning arrives
//!    halfway through.
//! 2. **Network drop:** a fresh cache steps through the same hours, and the network goes down for
//!    5 s partway through, failing new reads and the bodies of reads in flight.
//!
//! The "view" draws a step only after checking its values against the clean pass. A step whose
//! read fails is shown as an error and asked for again every second until it arrives.
//!
//! `s12 --listen 60` instead watches real memory-pressure notifications for 60 s and prints the
//! budget's limit as they arrive; drive it with `sudo memory_pressure -S -l warn` (and `-l normal`)
//! in another terminal.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use dv_data::budget::{Budget, Consumer};
use dv_data::prefetch::{Fetch, Prefetch};
use dv_data::pressure::{Level, Policy, PressureWatcher};
use dv_data::store::{FaultControl, FaultRates, FaultStore, RetryPolicy, RetryStore, http_store};
use dv_data::zarr::ZarrArray;
use dv_spike::era5::{self, URL};
use dv_spike::{Caches, Report};
use zarrs_object_store::object_store::ObjectStore;

const STEPS: u64 = 40;
const STEP_EVERY: Duration = Duration::from_millis(250);
const READ_AHEAD: u64 = 4;
const READ_PARALLEL: usize = 3;
const VIEW_RETRY_EVERY: Duration = Duration::from_secs(1);
/// A step that hasn't recovered by then counts as unrecovered.
const RECOVER_WITHIN: Duration = Duration::from_secs(20);
const DROP_AT_STEP: u64 = 12;
const DROP_FOR: Duration = Duration::from_secs(5);
/// Small enough that the cache is full (about 4.2 MB a step) by the time the pressure warning
/// arrives, so the warning has something to shed.
const BUDGET: u64 = 80 << 20;

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).is_some_and(|a| a == "--listen") {
        let secs = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(60);
        return listen(secs).await;
    }
    let start = args
        .get(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or_else(|| era5::fresh_hour(STEPS + 10));
    println!("first hour {start} (pass it as the argument to repeat this run)");
    let mut report = Report::new("S12", Caches::Warm);

    // The clean pass: what every step really holds.
    let clean = ZarrArray::open_http(URL, "/2m_temperature")
        .await
        .expect("open");
    let mut truth = HashMap::new();
    for s in 0..STEPS {
        let v = clean
            .chunk_f32(&[start + s, 0, 0])
            .await
            .expect("clean read");
        truth.insert(s, Arc::new(v));
    }
    let truth = Arc::new(truth);

    // Phase 1: random faults, and a pressure warning halfway.
    let rates = FaultRates {
        server_error: 0.20,
        truncate: 0.20,
        hang: 0.05,
    };
    let control = FaultControl::new(start);
    let (array, retry) = faulty(rates, control.clone()).await;
    let budget = Budget::new(BUDGET);
    let watcher = PressureWatcher::start(budget.clone(), Policy::default());
    let prefetch = Prefetch::new(
        budget.clone(),
        STEPS,
        READ_AHEAD,
        READ_PARALLEL,
        fetch(&array, start),
    );
    let mut pressure_note = String::new();
    let p1 = run_view(&prefetch, &truth, |s| {
        if s == STEPS / 2 {
            let before = budget.used_by(Consumer::Cache);
            watcher.apply(Level::Warning);
            pressure_note = format!(
                "cache {:.0} MB → {:.0} MB, limit {:.0} MB → {:.0} MB",
                before as f64 / 1e6,
                budget.used_by(Consumer::Cache) as f64 / 1e6,
                BUDGET as f64 / 1e6,
                budget.limit() as f64 / 1e6
            );
            assert!(
                budget.used() <= budget.limit(),
                "the cache shed to the new limit"
            );
        }
    })
    .await;
    watcher.apply(Level::Normal);
    let s = retry.stats();
    let injected = format!(
        "{} injected: {} × 503, {} truncated, {} hangs",
        control.injected(),
        control.server_errors.load(Ordering::Relaxed),
        control.truncations.load(Ordering::Relaxed),
        control.hangs.load(Ordering::Relaxed)
    );
    p1.report(
        &mut report,
        "faults",
        &format!("{injected}; retry layer: {s:?}"),
    );
    report.metric(
        "faults.pressure_warning",
        if budget.peak() <= BUDGET { 0.0 } else { 1.0 },
        "",
        Some(0.5),
        &format!("budget never over its limit; at the warning: {pressure_note}"),
    );
    drop(prefetch);

    // Phase 2: a network drop partway through, on a fresh cache.
    let control = FaultControl::new(start ^ 0xabc);
    let (array, retry) = faulty(FaultRates::default(), control.clone()).await;
    let prefetch = Prefetch::new(
        Budget::new(BUDGET),
        STEPS,
        READ_AHEAD,
        READ_PARALLEL,
        fetch(&array, start),
    );
    let c = control.clone();
    let p2 = run_view(&prefetch, &truth, move |s| {
        if s == DROP_AT_STEP {
            c.set_down(true);
            let c = c.clone();
            tokio::spawn(async move {
                tokio::time::sleep(DROP_FOR).await;
                c.set_down(false);
            });
        }
    })
    .await;
    let s = retry.stats();
    p2.report(
        &mut report,
        "network_drop",
        &format!(
            "network down {:?} from step {DROP_AT_STEP}; {} reads failed by the drop; retry layer: {s:?}",
            DROP_FOR,
            control.dropped.load(Ordering::Relaxed)
        ),
    );
    std::process::exit(if report.finish() { 0 } else { 1 });
}

async fn faulty(rates: FaultRates, control: Arc<FaultControl>) -> (ZarrArray, RetryStore) {
    let http = http_store(URL).expect("store");
    let faults = FaultStore::new(http, rates, control);
    let retry = RetryStore::new(
        Box::new(faults) as Box<dyn ObjectStore>,
        RetryPolicy::default(),
    );
    let array = ZarrArray::open_store(Box::new(retry.clone()), "/2m_temperature")
        .await
        .expect("open through faults");
    (array, retry)
}

fn fetch(array: &ZarrArray, start: u64) -> Fetch {
    let array = array.clone();
    Arc::new(move |step| {
        let array = array.clone();
        Box::pin(async move { array.chunk_f32(&[start + step, 0, 0]).await })
    })
}

/// What one phase's view saw.
#[derive(Debug, Default)]
struct Outcome {
    /// Time from asking for a step to drawing it, for steps that arrived on the first ask.
    drawn_first_try: Vec<Duration>,
    /// Time from asking to showing an error, for steps whose first ask failed.
    error_shown: Vec<Duration>,
    /// Time from the first error to drawing the step.
    recovered: Vec<Duration>,
    unrecovered: u64,
    wrong: u64,
}

impl Outcome {
    fn report(&self, report: &mut Report, phase: &str, note: &str) {
        report.metric(
            &format!("{phase}.wrong_values_drawn"),
            self.wrong as f64,
            "steps",
            Some(0.5),
            "every step's values are compared bit for bit with the clean pass before drawing",
        );
        report.metric(
            &format!("{phase}.unrecovered"),
            self.unrecovered as f64,
            "steps",
            Some(0.5),
            &format!(
                "{} steps showed an error and were asked for again every {:?}",
                self.error_shown.len(),
                VIEW_RETRY_EVERY
            ),
        );
        report.metric(
            &format!("{phase}.errors_shown"),
            self.error_shown.len() as f64,
            "steps",
            None,
            note,
        );
        if !self.drawn_first_try.is_empty() {
            report.samples(&format!("{phase}.step_drawn"), &self.drawn_first_try, None);
        }
        if !self.error_shown.is_empty() {
            report.samples(
                &format!("{phase}.error_shown_after"),
                &self.error_shown,
                None,
            );
            report.samples(&format!("{phase}.recovered_after"), &self.recovered, None);
        }
    }
}

/// Steps the view forward through every step, drawing each only once its values check out, and
/// asking again for steps that failed until they arrive.
async fn run_view(
    prefetch: &Prefetch,
    truth: &HashMap<u64, Arc<Vec<f32>>>,
    mut at_step: impl FnMut(u64),
) -> Outcome {
    let mut out = Outcome::default();
    // Steps showing an error: when the first error was shown, and when to ask again.
    let mut errors: HashMap<u64, (Instant, Instant)> = HashMap::new();
    let mut next = Instant::now();
    for step in 0..STEPS {
        tokio::time::sleep_until(next.into()).await;
        next += STEP_EVERY;
        at_step(step);
        prefetch.go_to(step, 1);
        let t = Instant::now();
        match prefetch.get(step).await {
            Ok(v) => {
                if !draw(&v, &truth[&step]) {
                    out.wrong += 1;
                }
                out.drawn_first_try.push(t.elapsed());
            }
            Err(e) => {
                out.error_shown.push(t.elapsed());
                println!("step {step}: shown as an error: {e}");
                errors.insert(step, (Instant::now(), Instant::now() + VIEW_RETRY_EVERY));
            }
        }
        retry_errors(prefetch, truth, &mut errors, &mut out).await;
    }
    // Keep asking for whatever is still an error, as a view on screen would.
    let deadline = Instant::now() + RECOVER_WITHIN;
    while !errors.is_empty() && Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(50)).await;
        retry_errors(prefetch, truth, &mut errors, &mut out).await;
    }
    out.unrecovered = errors.len() as u64;
    out
}

async fn retry_errors(
    prefetch: &Prefetch,
    truth: &HashMap<u64, Arc<Vec<f32>>>,
    errors: &mut HashMap<u64, (Instant, Instant)>,
    out: &mut Outcome,
) {
    let due: Vec<u64> = errors
        .iter()
        .filter(|(s, (_, again))| Instant::now() >= *again || prefetch.cached(**s).is_some())
        .map(|(s, _)| *s)
        .collect();
    for step in due {
        match prefetch.get(step).await {
            Ok(v) => {
                if !draw(&v, &truth[&step]) {
                    out.wrong += 1;
                }
                let (first, _) = errors.remove(&step).expect("listed");
                out.recovered.push(first.elapsed());
            }
            Err(_) => {
                errors.get_mut(&step).expect("listed").1 = Instant::now() + VIEW_RETRY_EVERY;
            }
        }
    }
}

/// "Draws" a step: true if its values are exactly the clean pass's.
fn draw(got: &[f32], want: &[f32]) -> bool {
    got.len() == want.len()
        && got
            .iter()
            .zip(want)
            .all(|(a, b)| a.to_bits() == b.to_bits())
}

async fn listen(secs: u64) {
    let budget = Budget::for_this_machine();
    let normal = budget.limit();
    let w = PressureWatcher::start(budget.clone(), Policy::default());
    println!(
        "listening for {secs} s (dispatch source {}); budget {} MB. In another terminal: sudo memory_pressure -S -l warn",
        if w.is_listening() {
            "connected"
        } else {
            "NOT connected"
        },
        normal / 1_000_000
    );
    let mut seen = 0;
    let end = Instant::now() + Duration::from_secs(secs);
    while Instant::now() < end {
        tokio::time::sleep(Duration::from_millis(100)).await;
        let h = w.history();
        for level in &h[seen..] {
            println!(
                "{level:?}: budget limit now {} MB",
                budget.limit() / 1_000_000
            );
        }
        seen = h.len();
    }
}
