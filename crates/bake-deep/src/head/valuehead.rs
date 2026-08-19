use burn::{Tensor, module::Module, nn::{Linear, LinearConfig}, tensor::Device};
use crate::head::VHead;

#[derive(Module, Debug)]
pub struct LinearValueHead {
    layer: Linear
}

impl LinearValueHead {
    pub fn new(d_input: usize, d_output: usize, device: &Device) -> Self {
        Self {
            layer: LinearConfig::new(d_input, d_output).init(device)
        }
    }
}

impl VHead for LinearValueHead {
    fn forward(&self, encoded: Tensor<2>) -> Tensor<1> {
        let x = self.layer.forward(encoded);
        x.squeeze_dim(1)
    }
}