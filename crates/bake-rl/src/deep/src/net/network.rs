//! A common trait for user-implemented networks
//! 
use burn::{module::ModuleDisplay, optim::{GradientsParams, ModuleOptimizer}, prelude::*};
use crate::data::Batchable;

/// A common trait for user-implemented networks
pub trait Network : Module + ModuleDisplay + Clone {
    /// the observation which the network takes
    type Obs: Batchable;
}

/// A helper update function (for bake-macro)
/// # Warning
/// - This function does not call `valid()` at the end
pub fn update<N: Network>(net: N, loss: Tensor<1>, lr: f64, opt: &mut ModuleOptimizer) -> N {
    let grads = loss.backward();
    let grads = GradientsParams::from_grads(grads, &net);
    opt.step(lr, net, grads)
}