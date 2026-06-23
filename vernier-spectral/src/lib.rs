//! # vernier-spectral
//!
//! The spectral detection pipeline (André et al. 2020/2021): from an uploaded
//! image buffer to the fitted phase planes that `vernier-pose` turns into a
//! [`Pose`](vernier_core::Pose).
//!
//! Everything here is generic over `B: ComputeBackend` and depends only on
//! `vernier-core`, so switching CPU for GPU is a one-line change in the CLI.
//!
//! Per pattern direction the chain is: forward FFT → isolate one spectral lobe
//! with a Gaussian band-pass (conjugate excluded) → inverse FFT → per-pixel
//! phase → unwrap → least-squares plane fit. The plane's constant term is the
//! sub-period phase; its gradients give orientation and peak location.
//!
//! - [`spectrum`] — FFT/filter/iFFT orchestration and per-direction result.
//! - [`planefit`] — unwrap + least-squares phase-plane fit (the resolution core).
//! - [`unwrap`] — 1D phase unwrapping used by the plane fit.

pub mod planefit;
pub mod spectrum;
pub mod unwrap;

pub use planefit::PhasePlane;
pub use spectrum::DirectionResult;
