//! Backend selection — the single place in the whole workspace allowed to name
//! concrete backends and choose between them.
//!
//! ## The dispatch problem, and the clean idiom for it
//!
//! [`ComputeBackend`](vernier_core::ComputeBackend) has an associated type
//! (`Buffer2D`), so it is not naively object-safe: you cannot
//! `Box<dyn ComputeBackend>` and swap at runtime without erasing the buffer
//! type, and erasing it would hurt the GPU path. So we keep the trait clean and
//! monomorphic and put the runtime choice here.
//!
//! The trick is that the unit of work must stay *generic over the backend* all
//! the way down — the benchmark uploads, FFTs, and detects, all of which touch
//! `B::Buffer2D`. A buffer-free trait object cannot express that. Instead we use
//! a trait with a **generic method**, [`BackendTask::run`]: the task is written
//! once against `B: ComputeBackend`, and [`dispatch`] calls it inside each match
//! arm with the concrete backend. Each arm monomorphizes the task for that
//! backend; the runtime branch is just picking which monomorphization to call.
//!
//! Adding a backend = one match arm here, and every task gains it for free.

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
