//! Categorical Head for policy-based methods
use burn::{Tensor, module::Module, nn::{Linear, LinearConfig}, tensor::Device};

use crate::{distribution::Categorical, head::Head, types::DiscreteMask};

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
    type Barrier = DiscreteMask;
    /// currently, fill_value is not used here
    fn forward(&self, encoded: Tensor<2>, barrier: Option<Self::Barrier>) -> Self::Output {
        let logits = self.layer.forward(encoded);
        Categorical::new(logits, barrier)
    }
}