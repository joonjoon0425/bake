//! A common trait for neural networks
//! 
//! 
use crate::core::data::WithBatchDim;
use burn::{module::ModuleDisplay, optim::{GradientsParams, ModuleOptimizer}, prelude::*};

/// A common trait for neural networks
pub trait Network : Module + ModuleDisplay + Clone {
    /// the observation which the network takes
    type Obs: WithBatchDim;
}

/// A helper update function (for bake-macro)
/// # Warning
/// - This function does not call `valid()` at the end
pub fn update<N: Network>(net: N, loss: Tensor<1>, lr: f64, opt: &mut ModuleOptimizer) -> N {
    let grads = loss.backward();
    let grads = GradientsParams::from_grads(grads, &net);
    opt.step(lr, net, grads)
}

pub mod layer;
pub mod sequential;