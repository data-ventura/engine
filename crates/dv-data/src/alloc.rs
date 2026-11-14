//! A global allocator that gives large blocks back to the system as soon as they're freed.
//!
//! The engine allocates column buffers, indexes and decoded chunks of hundreds of megabytes and
//! frees them as views change. macOS's allocator caches freed large blocks in the process, where
//! they still count towards its memory footprint and aren't reused for other sizes, so the footprint
//! only grows. This allocator maps every block of [`LARGE`] bytes or more straight from the kernel
//! and unmaps it when it's freed. Smaller blocks go to the system allocator.
//!
//! Binaries opt in with:
//!
//! ```ignore
//! #[global_allocator]
//! static ALLOC: dv_data::alloc::ReturningAlloc = dv_data::alloc::ReturningAlloc;
//! ```

use std::alloc::{GlobalAlloc, Layout, System};
use std::ptr;

/// Blocks of at least this many bytes are mapped from the kernel.
pub const LARGE: usize = 1 << 20;
/// The largest alignment a mapped block guarantees: the smallest page size on supported systems.
const PAGE: usize = 4096;

/// See the module docs.
#[derive(Debug, Clone, Copy, Default)]
pub struct ReturningAlloc;

fn is_large(layout: Layout) -> bool {
    layout.size() >= LARGE && layout.align() <= PAGE
}

fn map(size: usize) -> *mut u8 {
    // SAFETY: an anonymous private mapping touches no existing memory.
    let p = unsafe {
        libc::mmap(
            ptr::null_mut(),
            size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANON,
            -1,
            0,
        )
    };
    if p == libc::MAP_FAILED {
        ptr::null_mut()
    } else {
        p.cast()
    }
}

// SAFETY: large blocks are fresh mappings of at least the requested size, aligned to a page, and are
// unmapped with the same size they were mapped with, since `dealloc` and `realloc` receive the
// layout the block was allocated with. Everything else is forwarded to `System` unchanged.
unsafe impl GlobalAlloc for ReturningAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if is_large(layout) {
            map(layout.size())
        } else {
            // SAFETY: forwarded with the caller's guarantees.
            unsafe { System.alloc(layout) }
        }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if is_large(layout) {
            // Fresh anonymous mappings are zero-filled.
            map(layout.size())
        } else {
            // SAFETY: forwarded with the caller's guarantees.
            unsafe { System.alloc_zeroed(layout) }
        }
    }

    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        if is_large(layout) {
            // SAFETY: `p` was mapped by `map` with this size (see the impl's safety comment).
            unsafe { libc::munmap(p.cast(), layout.size()) };
        } else {
            // SAFETY: forwarded with the caller's guarantees.
            unsafe { System.dealloc(p, layout) }
        }
    }

    unsafe fn realloc(&self, p: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: the caller guarantees `new_size` with `layout`'s alignment is a valid layout.
        let new = unsafe { Layout::from_size_align_unchecked(new_size, layout.align()) };
        if !is_large(layout) && !is_large(new) {
            // SAFETY: forwarded with the caller's guarantees.
            return unsafe { System.realloc(p, layout, new_size) };
        }
        // SAFETY: `new` is a valid, non-zero layout; `p` holds `layout.size()` readable bytes and
        // is freed with the layout it was allocated with.
        unsafe {
            let q = self.alloc(new);
            if !q.is_null() {
                ptr::copy_nonoverlapping(p, q, layout.size().min(new_size));
                self.dealloc(p, layout);
            }
            q
        }
    }
}
