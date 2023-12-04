use crate::cuda::atomic::AtomicInt;
use crate::{BlockHeaderId, BlockVirtualId, DevicePointer};
use cuda_std::thread;

const EMPTY: u64 = u64::MAX;

fn hash(mut key: u64) -> u64 {
    key ^= key >> 16;
    key *= 0x85ebca6b;
    key ^= key >> 13;
    key *= 0xc2b2ae35;
    key ^= key >> 16;
    key
}

#[cfg_attr(not(target_os = "cuda"), derive(cust::DeviceCopy))]
#[derive(Copy, Clone, Debug, PartialEq, Eq, bytemuck::Zeroable)]
#[repr(C)]
pub struct GridHashMapEntry {
    pub key: BlockVirtualId,
    pub value: BlockHeaderId,
}

impl GridHashMapEntry {
    pub fn free() -> Self {
        Self {
            key: BlockVirtualId(EMPTY),
            value: BlockHeaderId(0),
        }
    }
}

#[cfg_attr(not(target_os = "cuda"), derive(cust::DeviceCopy))]
#[derive(Clone, Copy)]
#[repr(C)]
pub struct GridHashMap {
    entries: DevicePointer<GridHashMapEntry>,
    capacity: u32, // NOTE: must be a power of two.
}

impl GridHashMap {
    // The `capacity` must be a power of 2 and the `entries` buffer must have enough room for at least
    // `capacity` elements.
    pub unsafe fn from_raw_parts(entries: DevicePointer<GridHashMapEntry>, capacity: u32) -> Self {
        Self { entries, capacity }
    }

    pub fn insert_nonexistant_with(
        &mut self,
        key: BlockVirtualId,
        mut value: impl FnMut() -> BlockHeaderId,
    ) {
        let mut slot = hash(key.0) & (self.capacity as u64 - 1);

        // NOTE: if there is no more room in the hashmap to store the data, we just do nothing.
        // It is up to the user to detect the high occupancy, resize the hashmap, and re-run
        // the failed insertion.
        for _ in 0..self.capacity - 1 {
            let entry = unsafe { &mut *self.entries.as_mut_ptr().add(slot as usize) };
            let prev = unsafe { entry.key.0.global_atomic_cas(EMPTY, key.0) };
            if prev == EMPTY {
                entry.value = value();
                break;
            } else if prev == key.0 {
                break; // We found the key, and it already exists.
            }

            slot = (slot + 1) & (self.capacity as u64 - 1);
        }
    }

    pub fn get(&self, key: BlockVirtualId) -> Option<BlockHeaderId> {
        let mut slot = hash(key.0) & (self.capacity as u64 - 1);

        loop {
            let entry = unsafe { *self.entries.as_ptr().add(slot as usize) };
            if entry.key == key {
                return Some(entry.value);
            }

            if entry.key.0 == EMPTY {
                return None;
            }

            slot = (slot + 1) & (self.capacity as u64 - 1);
        }
    }
}

#[cfg_attr(target_os = "cuda", cuda_std::kernel)]
pub unsafe fn reset_hashmap(grid: GridHashMap) {
    let id = thread::index();
    if (id as u32) < grid.capacity {
        *grid.entries.as_mut_ptr().add(id as usize) = GridHashMapEntry::free();
    }
}
