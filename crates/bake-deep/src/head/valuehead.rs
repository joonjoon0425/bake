use burn::{Tensor, module::Module, nn::{Linear, LinearConfig}, tensor::Device};
use crate::types::ActionMask;
use crate::head::Head;

#[derive(Module, Debug)]
pub struct ValueHead {
    layer: Linear
}

impl ValueHead {
    pub fn new(d_input: usize, d_output: usize, device: &Device) -> Self {
        Self {
            layer: LinearConfig::new(d_input, d_output).init(device)
        }
    }
}

impl Head for ValueHead {
    type Output = Tensor<1>;

    fn forward<M: ActionMask<Value = Tensor<2>>>(&self, encoded: Tensor<2>, _: M, _: f32) -> Self::Output {
        let x = self.layer.forward(encoded);
        x.squeeze_dim(1)
    }
}