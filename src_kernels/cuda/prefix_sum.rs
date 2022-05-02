use cuda_std::{shared_array, thread};

#[cuda_std::kernel]
pub unsafe fn add_data_grp(data: *mut u32, data_len: u32, rhs: *mut u32) {
    let id = thread::index_1d();
    let bid = thread::block_idx_x();

    if id < data_len {
        *data.add(id as usize) += *rhs.add(bid as usize);
    }
}

// TODO: Benchmark and see if we can do 1024 instead of 512.
// TODO: use dynamic shared memory instead.
// TODO: optimize to avoid bank conflicts.
#[cuda_std::kernel]
pub unsafe fn prefix_sum_512(data: *mut u32, data_len: u32, aux: *mut u32) {
    const THREADS: usize = 512;
    let thread_id = thread::thread_idx_x();
    let block_id = thread::block_idx_x();
    let shared = shared_array![u32; THREADS];

    if block_id * THREADS as u32 >= data_len {
        return;
    }

    let data_block_len = data_len as usize - block_id as usize * THREADS;
    let shared_len = data_block_len.next_power_of_two().min(THREADS).max(1);
    let elt_id = (thread_id + block_id * THREADS as u32) as usize;

    prefix_sum(
        data,
        data_len as usize,
        elt_id,
        aux,
        shared,
        shared_len,
        thread_id,
        block_id,
    )
}

// NOTE:
//       `shared` must contain at least `shared_len` elements.
//       `shared_len` must be a power of two.
unsafe fn prefix_sum(
    data: *mut u32,
    data_len: usize,
    elt_id: usize,
    aux: *mut u32,
    shared: *mut u32,
    shared_len: usize,
    thread_id: u32,
    block_id: u32,
) {
    let bid = block_id as usize;
    let tid = thread_id as usize;

    // Init the shared memory.
    *shared.add(tid) = if elt_id < data_len {
        *data.add(elt_id)
    } else {
        0
    };

    // Up-Sweep.
    let mut d = shared_len / 2;
    let mut offset = 1;
    while d > 0 {
        thread::sync_threads();
        if tid < d {
            let ia = tid * 2 * offset + offset - 1;
            let ib = (tid * 2 + 1) * offset + offset - 1;

            let sum = *shared.add(ia) + *shared.add(ib);
            *shared.add(ib) = sum;
        }

        d /= 2;
        offset *= 2;
    }

    if tid == 0 {
        let total_sum = *shared.add(shared_len - 1);
        *aux.add(bid) = total_sum;
        *shared.add(shared_len - 1) = 0;
    }

    // Down-Sweep
    let mut d = 1;
    let mut offset = shared_len / 2;

    while d < shared_len {
        thread::sync_threads();
        if tid < d {
            let ia = tid * 2 * offset + offset - 1;
            let ib = (tid * 2 + 1) * offset + offset - 1;

            let a = *shared.add(ia);
            let b = *shared.add(ib);

            *shared.add(ia) = b;
            *shared.add(ib) = a + b;
        }

        d *= 2;
        offset /= 2;
    }

    // Writeback the result
    thread::sync_threads();
    if elt_id < data_len as usize {
        *data.add(elt_id) = *shared.add(tid);
    };
}
