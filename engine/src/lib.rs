//! lfm2-engine: Rust-host ash/Vulkan inference engine for LFM2.5-8B-A1B.
//!
//! Implements the LFM2 (MoE + short-convolution hybrid) forward graph directly
//! in Rust. Tensor math is dispatched to the W9100 over Vulkan compute (ash);
//! a pure-Rust CPU reference implementation of the same arithmetic is kept for
//! testing and for the first-hardware Q6_K multiply comparison.

mod attention_capture;
pub mod gpu_lfm;
pub mod kernels;
pub mod lfm;
pub mod model_file;
pub mod ops;
pub mod q6k;
pub mod text_output;
pub mod vulkan;
pub mod work_bound;
#[cfg(test)]
mod scale_pair_tests;

pub use lfm::Hyperparams;
pub use model_file::{ModelFile, TensorInfo};
