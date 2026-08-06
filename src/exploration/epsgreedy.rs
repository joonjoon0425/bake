use burn::{Tensor, tensor::{Bool, ElementConversion, backend::Backend}};
use rand::{RngExt, SeedableRng, rngs::StdRng};

pub struct EpsGreedy {
    eps: f32,
    rng: StdRng
}

impl EpsGreedy {
    pub fn new(eps: f32, seed: u64) -> Self { EpsGreedy { eps, rng: StdRng::seed_from_u64(seed) } }
    pub fn eps(&self) -> f32 { self.eps }
    pub fn eps_mut(&mut self) -> &mut f32 { &mut self.eps }

    pub fn select_action<B: Backend>(&mut self, qvalues: Tensor<B, 1>) -> i64 {
        if self.rng.random_range(0.0..1.0f32) < self.eps {
            return self.rng.random_range(0..qvalues.shape()[0]) as i64
        }
        qvalues.argmax(0).try_into_scalar().unwrap().elem()
    }

    pub fn select_action_masked<B: Backend>(&mut self, qvalues: Tensor<B, 1>, mask: &[bool]) -> i64 {
        debug_assert!(qvalues.shape()[0] == mask.len(), "the length of qvalues and length of mask does not equals");
        
        let range = mask.iter().filter(|&&ok| ok).count();
        if self.rng.random_range(0.0..1.0f32) < self.eps {
            let r = self.rng.random_range(0..range);
            let action = mask.iter().enumerate().filter(|(_, ok)| **ok).nth(r).unwrap().0 as i64;
            return action;
        }

        let qvalues: Vec<f32> = qvalues.into_data().into_vec().unwrap();
        let action = qvalues.iter()
            .enumerate().filter(|(i, _)| mask[*i])
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap().0 as i64;
        action
    }
}