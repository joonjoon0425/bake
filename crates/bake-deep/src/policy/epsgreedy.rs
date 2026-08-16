//! epsilon-greedy policy
//! 

use burn::{Tensor, tensor::{Distribution, Int}};
use rand::{RngExt, SeedableRng, rngs::StdRng};

use crate::types::ActionMask;
pub struct EpsGreedy {
    eps: f32,
    rng: StdRng,
}

impl EpsGreedy {
    /// Create a new EpsGreedy policy
    pub fn new(seed: u64, eps: f32) -> Self {
        Self {
            eps,
            rng: StdRng::seed_from_u64(seed)
        }
    }

    pub fn eps(&self) -> f32 { self.eps }
    pub fn eps_mut(&mut self) -> &mut f32 { &mut self.eps }

    /// sample an action from given Q values.
    pub fn sample<M: ActionMask<Value = Tensor<2>>>(&mut self, qvalues: Tensor<2>, mask: M) -> Tensor<1, Int> {
        if self.rng.random_range(0.0..1.0) < self.eps {
            let random = Tensor::random_like(&qvalues, Distribution::Default);    
            return mask.apply(random, -1f32).argmax(1).squeeze_dim(0);
        } else {
            return mask.apply(qvalues, -1e9).argmax(1).squeeze_dim(0);
        }
        
    }
}