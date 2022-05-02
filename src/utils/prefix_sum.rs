use cust::{
    error::CudaResult,
    memory::{CopyDestination, DeviceBuffer},
    prelude::*,
};

// TODO: should this be moved to the same file as the kernel code?
struct PrefixSumStage {
    len: u32, // NOTE: this MUST be an u32 to match the kernel’s input type.
    capacity: u32,
    buffer: DeviceBuffer<u32>,
}

pub struct PrefixSumWorkspace {
    stages: Vec<PrefixSumStage>,
    num_stages: usize,
}

impl PrefixSumWorkspace {
    const THREADS: u32 = 512;

    pub fn new() -> Self {
        Self {
            stages: vec![],
            num_stages: 0,
        }
    }

    pub fn with_capacity(buffer_len: u32) -> CudaResult<Self> {
        let mut result = Self {
            stages: vec![],
            num_stages: 0,
        };
        result.reserve(buffer_len)?;
        Ok(result)
    }

    pub fn reserve(&mut self, buffer_len: u32) -> CudaResult<()> {
        let mut stage_len = buffer_len / Self::THREADS + ((buffer_len % Self::THREADS) != 0) as u32;

        if self.stages.is_empty() || self.stages[0].capacity < stage_len {
            // Reinitialize the auxiliary buffers.
            self.stages.clear();

            while stage_len != 1 {
                let buffer = DeviceBuffer::zeroed(stage_len as usize)?;
                self.stages.push(PrefixSumStage {
                    len: stage_len,
                    capacity: stage_len,
                    buffer,
                });

                stage_len = stage_len / Self::THREADS + ((stage_len % Self::THREADS) != 0) as u32;
            }

            // The last stage always has only 1 element.
            self.stages.push(PrefixSumStage {
                len: 1,
                capacity: 1,
                buffer: DeviceBuffer::zeroed(1)?,
            });
            self.num_stages = self.stages.len();
        } else if self.stages[0].len != stage_len {
            // The stages have big enough buffers, but we need to adjust their length.
            self.num_stages = 0;
            while stage_len != 1 {
                self.stages[self.num_stages].len = stage_len;
                self.num_stages += 1;
                stage_len = stage_len / Self::THREADS + ((stage_len % Self::THREADS) != 0) as u32;
            }

            // The last stage always has only 1 element.
            self.stages[self.num_stages].len = 1;
            self.num_stages += 1;
        }

        Ok(())
    }

    pub fn read_max_scan_value(&mut self) -> cust::error::CudaResult<u32> {
        for stage in &self.stages {
            if stage.len == 1 {
                // This is the last stage, it contains the total sum.
                let mut value = [0u32];
                stage.buffer.index(0).copy_to(&mut value)?;
                return Ok(value[0]);
            }
        }

        panic!("The GPU prefix sum has not been initialized yet.")
    }

    pub fn launch(
        &mut self,
        module: &Module,
        stream: &Stream,
        buffer: &mut DeviceBuffer<u32>,
        buffer_len: u32, // NOTE: this MUST be an u32 to match the kernel’s input type.
    ) -> cust::error::CudaResult<()> {
        // If this fails, the kernel launches bellow must be changed because we are using
        // a fixed size for the shared memory currently.
        assert_eq!(Self::THREADS, 512);

        self.reserve(buffer_len)?;

        let ngroups0 = self.stages[0].len;
        let aux0 = self.stages[0].buffer.as_device_ptr();
        unsafe {
            launch!(
                module.prefix_sum_512<<<ngroups0, Self::THREADS, 0, stream>>>(buffer.as_device_ptr(), buffer_len, aux0)
            )?;
        }

        for i in 0..self.num_stages - 1 {
            let len: u32 = self.stages[i + 0].len;
            let ngroups = self.stages[i + 1].len;
            let buf = self.stages[i + 0].buffer.as_device_ptr();
            let aux = self.stages[i + 1].buffer.as_device_ptr();

            unsafe {
                launch!(
                    module.prefix_sum_512<<<ngroups, Self::THREADS, 0, stream>>>(buf, len, aux)
                )?;
            }
        }

        if self.num_stages > 2 {
            for i in (0..self.num_stages - 2).rev() {
                let len: u32 = self.stages[i + 0].len;
                let ngroups = self.stages[i + 1].len;
                let buf = self.stages[i + 0].buffer.as_device_ptr();
                let aux = self.stages[i + 1].buffer.as_device_ptr();

                unsafe {
                    launch!(
                        module.add_data_grp<<<ngroups, Self::THREADS, 0, stream>>>(buf, len, aux)
                    )?;
                }
            }
        }

        if self.num_stages > 1 {
            unsafe {
                launch!(
                    module.add_data_grp<<<ngroups0, Self::THREADS, 0, stream>>>(buffer.as_device_ptr(), buffer_len, aux0)
                )?;
            }
        }

        Ok(())
    }
}

#[allow(dead_code)]
mod tests {
    use super::PrefixSumWorkspace;
    use cust::prelude::{CopyDestination, DeviceBuffer, Module, Stream};

    fn test_prefix_sum(module: &Module, stream: &Stream) {
        let mut rng = oorandom::Rand32::new(42);

        const THREADS: u32 = 512;
        let mut n = 100_000_000;
        let mut psum = PrefixSumWorkspace::new();

        while n >= 1 {
            let mut buffer: Vec<_> = (0..n).map(|_| rng.rand_u32() % 100).collect();
            let t0 = instant::now();
            let naive = naive_prefix_sum(&buffer);
            info!("Naive time n = {}: {}", n, instant::now() - t0);

            let mut gpu_buffer = DeviceBuffer::from_slice(&buffer).unwrap();
            psum.reserve(n).unwrap();

            let t0 = instant::now();
            psum.launch(module, stream, &mut gpu_buffer, n).unwrap();
            stream.synchronize().unwrap();
            info!("GPU time: {}", instant::now() - t0);

            gpu_buffer.copy_to(&mut buffer).unwrap();
            for i in 0..buffer.len() {
                if buffer[i] != naive[i] {
                    info!(
                        ">>>>>>>>>>>>>> FAILED at {} {} vs. {}",
                        i, buffer[i], naive[i]
                    );
                    break;
                }
            }

            n /= 10;
        }
    }

    fn naive_prefix_sum(elts: &[u32]) -> Vec<u32> {
        let mut result = elts.to_vec();
        for i in 1..elts.len() {
            result[i] = result[i - 1] + result[i];
        }
        result.insert(0, 0);
        result.pop();
        result
    }
}
