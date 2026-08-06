use burn::{Tensor, module::Module, nn::{Linear, LinearConfig}, tensor::backend::{AutodiffBackend, Backend}};

use crate::encoderhead::Head;

#[derive(Module, Debug)]
pub struct LinearHead<B: Backend> {
    linear: Linear<B>,
}

impl<B: AutodiffBackend> LinearHead<B> {
    pub fn new(d_input: usize, d_output: usize, device: &B::Device) -> Self {
        Self {
            linear: LinearConfig::new(d_input, d_output).init(device)
        }
    }
}

impl<B: AutodiffBackend, const D: usize> Head<B, D> for LinearHead<B> {
    type Output = Tensor<B, D>;

    fn forward(&self, encoded: Tensor<B, D>) -> Self::Output {
        self.linear.forward(encoded)
    }
}