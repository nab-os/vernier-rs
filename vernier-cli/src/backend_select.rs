//! Backend selection — the one place allowed to name concrete backends and
//! choose between them at runtime.
//!
//! [`ComputeBackend`](vernier_core::ComputeBackend) has an associated `Buffer2D`
//! type, so it isn't object-safe — you can't `Box<dyn ComputeBackend>` and swap
//! at runtime without erasing the buffer type. Instead the unit of work stays
//! generic: [`BackendTask::run`] is a generic method written once against
//! `B: ComputeBackend`, and [`dispatch`] calls it in each match arm with the
//! concrete backend. Adding a backend is one arm here, and every task gets it.

use vernier_core::ComputeBackend;
use vernier_cpu::CpuBackend;
#[cfg(feature = "gpu")]
use vernier_gpu::GpuBackend;
#[cfg(feature = "cuda")]
use vernier_cuda::CudaBackend;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    Cpu,
    #[cfg(feature = "gpu")]
    Gpu,
    #[cfg(feature = "cuda")]
    Cuda,
}

impl BackendKind {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "cpu" => Some(Self::Cpu),
            #[cfg(feature = "gpu")]
            "gpu" => Some(Self::Gpu),
            #[cfg(feature = "cuda")]
            "cuda" => Some(Self::Cuda),
            _ => None,
        }
    }

    pub fn hint() -> &'static str {
        #[cfg(all(feature = "gpu", feature = "cuda"))]
        return "cpu, gpu, cuda";
        #[cfg(all(feature = "gpu", not(feature = "cuda")))]
        return "cpu, gpu";
        #[cfg(not(feature = "gpu"))]
        return "cpu";
    }
}

pub trait BackendTask {
    type Output;
    fn run<B: ComputeBackend>(&self, backend: &B) -> Self::Output;
}

pub fn dispatch<T: BackendTask>(kind: BackendKind, task: &T) -> T::Output {
    match kind {
        BackendKind::Cpu => task.run(&CpuBackend::new()),
        #[cfg(feature = "gpu")]
        BackendKind::Gpu => task.run(&GpuBackend::new()),
        #[cfg(feature = "cuda")]
        BackendKind::Cuda => task.run(&CudaBackend::new().expect("CUDA init failed")),
    }
}
