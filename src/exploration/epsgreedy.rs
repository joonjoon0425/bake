use burn::{Tensor, tensor::{Bool, ElementConversion, backend::Backend}};

pub struct EpsGreedy {
    eps: f32,
}

impl EpsGreedy {
    pub fn new(eps: f32) -> Self { EpsGreedy { eps } }
    pub fn eps(&self) -> f32 { self.eps }
    pub fn eps_mut(&mut self) -> &mut f32 { &mut self.eps }

    pub fn select_action<B: Backend>(&self, qvalues: Tensor<B, 1>) -> i64 {
        qvalues.argmax(1).try_into_scalar().unwrap().elem()
    }

    pub fn select_action_masked<B: Backend>(&self, qvalues: Tensor<B, 1>, mask: Tensor<B, 1, Bool>) -> i64 {
        let qvalues = qvalues.mask_fill(mask, -1e+9);
        qvalues.argmax(1).try_into_scalar().unwrap().elem()
    }
}