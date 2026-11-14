//! The returning allocator keeps contents across reallocations and gives large blocks back.

use dv_data::alloc::{LARGE, ReturningAlloc};

#[global_allocator]
static ALLOC: ReturningAlloc = ReturningAlloc;

#[test]
fn contents_survive_growing_and_shrinking_across_the_threshold() {
    let mut v: Vec<u32> = Vec::new();
    let n = (3 * LARGE / 4) as u32;
    for i in 0..n {
        v.push(i);
    }
    assert!(v.iter().enumerate().all(|(i, &x)| x == i as u32));
    v.truncate(100);
    v.shrink_to_fit();
    assert_eq!(v, (0..100).collect::<Vec<u32>>());
    let z = vec![0u64; LARGE];
    assert!(z.iter().all(|&x| x == 0));
}

#[cfg(target_os = "macos")]
#[test]
fn large_blocks_leave_the_footprint_when_freed() {
    fn footprint() -> u64 {
        // SAFETY: proc_pid_rusage writes a rusage_info_v4 into the buffer we pass.
        unsafe {
            let mut info = std::mem::zeroed::<libc::rusage_info_v4>();
            libc::proc_pid_rusage(
                libc::getpid(),
                libc::RUSAGE_INFO_V4,
                (&raw mut info).cast::<libc::rusage_info_t>(),
            );
            info.ri_phys_footprint
        }
    }
    let before = footprint();
    let v = vec![1u8; 256 << 20];
    assert!(footprint() > before + (200 << 20));
    drop(v);
    assert!(
        footprint() < before + (16 << 20),
        "freed block still counted"
    );
}
