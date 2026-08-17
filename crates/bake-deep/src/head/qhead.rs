use burn::{Tensor, module::Module, nn::{Linear, LinearConfig}, tensor::Device};
use crate::types::ActionMask;
use crate::head::Head;

#[derive(Module, Debug)]
pub struct QHead {
    layer: Linear
}

impl QHead {
    pub fn new(d_input: usize, d_output: usize, device: &Device) -> Self {
        Self {
            layer: LinearConfig::new(d_input, d_output).init(device)
        }
    }
}

impl Head for QHead {
    type Output = Tensor<2>;

    fn forward<M: ActionMask<Value = Tensor<2>>>(&self, encoded: Tensor<2>, mask: M, fill_value: f32) -> Self::Output {
        let x = self.layer.forward(encoded);
        mask.apply(x, fill_value)
    }
}

#[derive(Module, Debug)]
pub struct DuelingQHead {
    value: Linear,
    advantage: Linear,
}

impl DuelingQHead {
    pub fn new(d_input: usize, d_output: usize, device: &Device) -> Self {
        Self {
            value: LinearConfig::new(d_input, 1).init(device),
            advantage: LinearConfig::new(d_input, d_output).init(device),
        }
    }
}

impl Head for DuelingQHead {
    type Output = Tensor<2>;

    fn forward<M: ActionMask<Value = Tensor<2>>>(&self, encoded: Tensor<2>, mask: M, fill_value: f32) -> Self::Output {
        let v = self.value.forward(encoded.clone());
        let a = self.advantage.forward(encoded);
        let mean = mask.clone().mean_dim(1, a.clone());
        mask.apply(v + (a - mean), fill_value)
    }
}