//! epsilon-greedy policy
//! 

use burn::{Tensor, tensor::Distribution};
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

    /// sample an action from given Q values.<br>Currently, it pulls down the Tensor qvalues to raw float vector on cpu.
    /// other implementations can be considered...
    // pub fn sample<const D: usize>(&mut self, qvalues: Tensor<1>, mask: DiscreteMask<D>) -> i64 {
    //     if self.rng.random_range(0.0..1.0) < self.eps {
    //         let n_possible_actions = mask.n_possible_actions();
    //         let r = self.rng.random_range(0..n_possible_actions);
    //         let action = mask.possible_actions().nth(r).unwrap();
    //         action as i64
    //     } else {
    //         let qvalues: Vec<f32> = qvalues.into_data().into_vec().unwrap();
    //         let mut qmax = f32::MIN;
    //         let mut candidate = 0;
    //         for action in mask.possible_actions() {
    //             let qvalue = qvalues[action];
    //             if qmax < qvalue {
    //                 qmax = qvalue;
    //                 candidate = action;
    //             }
    //         }
    //         candidate as i64
    //     }
    // }

    /// sample an action from given Q values.
    pub fn sample<M: ActionMask<Value = Tensor<1>>>(&mut self, qvalues: Tensor<1>, mask: M) -> i64 {
        if self.rng.random_range(0.0..1.0) < self.eps {
            let random = Tensor::random_like(&qvalues, Distribution::Default);    
            return mask.apply(random, -1f32).argmax(0).into_scalar();
        } else {
            return mask.apply(qvalues, -1e9).argmax(0).into_scalar();
        }
        
    }
}