//! Categorical Head for policy-based methods
use burn::{Tensor, module::Module, nn::{Linear, LinearConfig}, tensor::Device};

use crate::{distribution::Categorical, head::Head, types::ActionMask};

#[derive(Module, Debug)]
pub struct CategoricalHead {
    layer: Linear,
}

impl CategoricalHead {
    pub fn new(d_input: usize, d_output: usize, device: &Device) -> Self {
        Self {
            layer: LinearConfig::new(d_input, d_output).init(device)
        }
    }
}

impl Head for CategoricalHead {
    type Output = Categorical;

    /// currently, fill_value is not used here
    fn forward<M: ActionMask<Value = Tensor<2>>>(&self, encoded: Tensor<2>, mask: M, _: f32) -> Self::Output {
        let logits = self.layer.forward(encoded);
        Categorical::new(logits, mask)
    }
}