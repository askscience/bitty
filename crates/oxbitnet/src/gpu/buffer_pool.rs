use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use wgpu::BufferUsages;

/// A shareable reference to a GPU buffer.
pub type GpuBuf = Arc<wgpu::Buffer>;

/// GPU buffer pool with size-bucketed reuse.
///
/// Maintains a small per-bucket pool of free buffers to avoid
/// repeated GPU allocations during autoregressive generation.
const MAX_POOLED_PER_BUCKET: usize = 8;

pub struct BufferPool {
    device: Arc<wgpu::Device>,
    alignment: u64,
    free: Mutex<Vec<VecDeque<GpuBuf>>>,
    buckets: Vec<u64>,
}

impl BufferPool {
    pub fn new(device: Arc<wgpu::Device>, alignment: u64) -> Self {
        let buckets: Vec<u64> = (6..=26).map(|s| 1u64 << s).collect();
        let free = Mutex::new(vec![VecDeque::new(); buckets.len()]);
        Self {
            device,
            alignment,
            free,
            buckets,
        }
    }

    fn align_size(&self, size: u64) -> u64 {
        size.div_ceil(self.alignment) * self.alignment
    }

    fn bucket_index(&self, aligned: u64) -> Option<usize> {
        self.buckets.binary_search(&aligned).ok()
    }

    /// Obtain a buffer of at least `size` bytes.
    /// Attempts to reuse a pooled buffer before allocating new.
    pub fn acquire(&self, size: u64, usage: BufferUsages) -> GpuBuf {
        let aligned = self.align_size(size.max(4));
        if let Some(idx) = self.bucket_index(aligned) {
            let mut free = self.free.lock().unwrap();
            let slot = &mut free[idx];
            while let Some(candidate) = slot.pop_back() {
                if candidate.usage() == usage && candidate.size() >= aligned {
                    return candidate;
                }
            }
        }
        Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: aligned,
            usage,
            mapped_at_creation: false,
        }))
    }

    /// Release a buffer back to the pool for future reuse.
    pub fn release(&self, buffer: GpuBuf) {
        let aligned = self.align_size(buffer.size());
        if let Some(idx) = self.bucket_index(aligned) {
            let mut free = self.free.lock().unwrap();
            let slot = &mut free[idx];
            if slot.len() < MAX_POOLED_PER_BUCKET {
                slot.push_back(buffer);
            }
        }
    }
}
