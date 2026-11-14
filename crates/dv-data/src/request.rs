//! Generations and latest-wins scheduling for a view's requests.
//!
//! Every request carries the generation of the view that asked for it, and every new interaction
//! advances the generation. Each view has at most one request running and one waiting: a newer
//! request replaces the waiting one, and the running one is cancelled by dropping its future, which
//! drops its object-store reads. A result whose generation is no longer current is discarded, never
//! delivered, so a stale frame can't be drawn.

use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use tokio::sync::watch;

/// The generation a request was made in. Cheap to clone.
#[derive(Debug, Clone)]
pub struct Generation {
    value: u64,
    current: Arc<AtomicU64>,
}

/// A request whose generation has passed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("cancelled: superseded by a newer request")]
pub struct Cancelled;

impl Generation {
    pub fn value(&self) -> u64 {
        self.value
    }

    /// Whether no newer request has been made since this one.
    pub fn is_current(&self) -> bool {
        self.current.load(Ordering::Acquire) == self.value
    }

    /// For long CPU work to call between steps: `Err` once the request is obsolete.
    pub fn check(&self) -> Result<(), Cancelled> {
        if self.is_current() {
            Ok(())
        } else {
            Err(Cancelled)
        }
    }
}

/// The view's side: submits requests, each in a new generation.
#[derive(Debug)]
pub struct Submitter<R> {
    current: Arc<AtomicU64>,
    tx: watch::Sender<Option<(Generation, R)>>,
}

/// The worker's side: see [`serve`].
#[derive(Debug)]
pub struct Requests<R> {
    rx: watch::Receiver<Option<(Generation, R)>>,
}

/// A connected submitter and worker side for one view.
pub fn channel<R>() -> (Submitter<R>, Requests<R>) {
    let (tx, rx) = watch::channel(None);
    (
        Submitter {
            current: Arc::new(AtomicU64::new(0)),
            tx,
        },
        Requests { rx },
    )
}

impl<R> Submitter<R> {
    /// Starts a new generation for `request`. It replaces any waiting request, and a running one
    /// is cancelled.
    pub fn submit(&self, request: R) -> Generation {
        let value = self.current.fetch_add(1, Ordering::AcqRel) + 1;
        let g = Generation {
            value,
            current: self.current.clone(),
        };
        self.tx.send_replace(Some((g.clone(), request)));
        g
    }

    /// The generation of the latest request, for checking results drawn elsewhere.
    pub fn current(&self) -> u64 {
        self.current.load(Ordering::Acquire)
    }
}

/// How a view's worker treats requests that arrive quickly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Policy {
    /// Wait until no newer request has arrived for this long before starting one. Zero starts
    /// at once.
    pub settle: Duration,
    /// Cancel a running request as soon as a newer one arrives. Otherwise it runs to the end and
    /// its result is discarded if stale.
    pub cancel_running: bool,
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            settle: Duration::ZERO,
            cancel_running: true,
        }
    }
}

/// Runs a view's requests until its [`Submitter`] is dropped: always the latest request, one at a
/// time, with the default [`Policy`].
pub async fn serve<R, T, F, Fut>(requests: Requests<R>, run: F, deliver: impl FnMut(Generation, T))
where
    R: Clone,
    F: FnMut(Generation, R) -> Fut,
    Fut: Future<Output = T>,
{
    serve_with(requests, Policy::default(), run, deliver).await;
}

/// Runs a view's requests until its [`Submitter`] is dropped: always the latest request, one at a
/// time. A request that's superseded while running is cancelled by dropping its future if the
/// policy says so. Results go to `deliver` only if their generation is still current when they
/// finish.
pub async fn serve_with<R, T, F, Fut>(
    mut requests: Requests<R>,
    policy: Policy,
    mut run: F,
    mut deliver: impl FnMut(Generation, T),
) where
    R: Clone,
    F: FnMut(Generation, R) -> Fut,
    Fut: Future<Output = T>,
{
    loop {
        if requests.rx.changed().await.is_err() {
            return;
        }
        // Each pass takes the latest request; one superseded loops straight back here with the
        // newer one already marked as seen-but-unhandled.
        loop {
            if !policy.settle.is_zero() {
                loop {
                    match tokio::time::timeout(policy.settle, requests.rx.changed()).await {
                        Err(_) => break,
                        Ok(Err(_)) => return,
                        Ok(Ok(())) => {}
                    }
                }
            }
            let Some((g, r)) = requests.rx.borrow_and_update().clone() else {
                break;
            };
            if !policy.cancel_running {
                let out = run(g.clone(), r).await;
                if g.is_current() {
                    deliver(g, out);
                }
                if requests.rx.has_changed().unwrap_or(false) {
                    continue;
                }
                break;
            }
            tokio::select! {
                out = run(g.clone(), r) => {
                    if g.is_current() {
                        deliver(g, out);
                    }
                    break;
                }
                changed = requests.rx.changed() => {
                    if changed.is_err() {
                        return;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::sync::atomic::AtomicUsize;

    use super::*;

    #[tokio::test(start_paused = true)]
    async fn latest_wins_and_nothing_stale_is_delivered() {
        let (tx, rx) = channel::<u32>();
        let running = Arc::new(AtomicUsize::new(0));
        let most = Arc::new(AtomicUsize::new(0));
        let delivered = Arc::new(Mutex::new(Vec::new()));
        let worker = tokio::spawn({
            let (running, most, delivered) = (running.clone(), most.clone(), delivered.clone());
            serve(
                rx,
                move |g, step: u32| {
                    let (running, most) = (running.clone(), most.clone());
                    async move {
                        let n = running.fetch_add(1, Ordering::SeqCst) + 1;
                        most.fetch_max(n, Ordering::SeqCst);
                        // Decrements even when the future is dropped mid-sleep.
                        struct Done(Arc<AtomicUsize>);
                        impl Drop for Done {
                            fn drop(&mut self) {
                                self.0.fetch_sub(1, Ordering::SeqCst);
                            }
                        }
                        let _done = Done(running);
                        tokio::time::sleep(Duration::from_millis(30)).await;
                        (g.value(), step)
                    }
                },
                move |g, out| {
                    assert!(g.is_current());
                    delivered.lock().unwrap().push(out);
                },
            )
        });
        // 50 steps 10 ms apart, each faster than a request takes.
        for step in 0..50 {
            tx.submit(step);
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(
            most.load(Ordering::SeqCst),
            1,
            "one request running at a time"
        );
        assert_eq!(*delivered.lock().unwrap(), vec![(50, 49)], "only the last");
        // Once requests are spaced out, every one is delivered.
        for step in 50..53 {
            tx.submit(step);
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert_eq!(delivered.lock().unwrap().len(), 4);
        drop(tx);
        worker.await.unwrap();
    }

    #[test]
    fn generations_go_stale() {
        let (tx, _rx) = channel::<()>();
        let a = tx.submit(());
        assert!(a.check().is_ok());
        let b = tx.submit(());
        assert_eq!(a.check(), Err(Cancelled));
        assert!(b.is_current());
        assert_eq!(tx.current(), b.value());
    }
}
