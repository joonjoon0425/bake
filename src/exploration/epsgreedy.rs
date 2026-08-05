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

    pub fn select_action_masked<B: Backend>(&self, qvalues: Tensor<B, 1>, mask: Tensor<B, 1, Bool>) -> i64 {
        let qvalues = qvalues.mask_fill(mask, -1e+9);
        // TODO: FIX HERE SO THAT THE MASKED VALUES ARE NOT SELECTED.
        qvalues.argmax(0).try_into_scalar().unwrap().elem()
    }
}