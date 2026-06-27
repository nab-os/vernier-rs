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
use vernier_gpu::GpuBackend;
#[cfg(feature = "cuda")]
use vernier_cuda::CudaBackend;

/// Which compute backend to run with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    /// The `rustfft`/`ndarray` reference backend.
    Cpu,
    /// The Vulkano GPU backend.
    Gpu,
    /// The CUDA backend (requires `--features cuda`).
    #[cfg(feature = "cuda")]
    Cuda,
}

impl BackendKind {
    /// Parses a backend name from the command line.
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "cpu" => Some(Self::Cpu),
            "gpu" => Some(Self::Gpu),
            #[cfg(feature = "cuda")]
            "cuda" => Some(Self::Cuda),
            _ => None,
        }
    }
}

/// A unit of work that can run against *any* backend.
///
/// The generic method is the whole point: `run` is written once and the
/// dispatcher supplies the concrete `B`. Implementors (commands) hold their
/// inputs as fields and consume them in `run`.
pub trait BackendTask {
    /// The task's output type.
    type Output;

    /// Executes the task against a concrete backend `B`.
    fn run<B: ComputeBackend>(&self, backend: &B) -> Self::Output;
}

/// Instantiates the backend chosen by `kind` and runs `task` against it.
///
/// The single place that maps a runtime [`BackendKind`] to a concrete type. New
/// backend => new arm; nothing else changes.
pub fn dispatch<T: BackendTask>(kind: BackendKind, task: &T) -> T::Output {
    match kind {
        BackendKind::Cpu => task.run(&CpuBackend::new()),
        BackendKind::Gpu => task.run(&GpuBackend::new()),
        #[cfg(feature = "cuda")]
        BackendKind::Cuda => task.run(&CudaBackend::new().expect("CUDA init failed")),
    }
}
