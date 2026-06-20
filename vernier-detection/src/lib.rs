//! # vernier-detection
//!
//! The spectral detection pipeline (André et al. 2020/2021): from an uploaded
//! image buffer to the fitted phase planes that `vernier-pose` turns into a
//! [`Pose`](vernier_core::Pose).
//!
//! ## Backend-generic by construction
//!
//! Every function here is generic over `B: ComputeBackend`. This crate depends
//! *only* on `vernier-core` — never on `vernier-cpu` or `vernier-gpu`. If this
//! crate compiles, the [`ComputeBackend`](vernier_core::ComputeBackend) surface
//! is sufficient to express detection, and swapping CPU for GPU is a one-line
//! change in `vernier-cli`.
//!
//! ## The real processing chain
//!
//! Per pattern direction: forward FFT -> isolate one spectral lobe with a
//! Gaussian band-pass (conjugate excluded) -> inverse FFT -> per-pixel phase ->
//! unwrap -> least-squares plane fit. The plane's constant term is the
//! high-resolution sub-period phase; its gradients give orientation and peak
//! location. Fitting across all pixels — not reading one bin — is what earns the
//! sub-1/1000-pixel resolution.
//!
//! ## Stages
//!
//! - [`spectrum`] — the FFT/filter/iFFT orchestration and per-direction result.
//! - [`planefit`] — unwrap + least-squares phase-plane fit (the resolution core).
//! - [`unwrap`] — 1D phase unwrapping used by the plane fit.

pub mod planefit;
pub mod spectrum;
pub mod unwrap;

pub use planefit::PhasePlane;
pub use spectrum::DirectionResult;
