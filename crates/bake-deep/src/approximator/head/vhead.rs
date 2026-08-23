use burn::{Tensor, module::{AutodiffModule, Module, ModuleDisplay}, nn::{Linear, LinearConfig}, tensor::Device};

pub trait VHead: AutodiffModule + Clone + ModuleDisplay {
    fn forward(&self, encoded: Tensor<2>) -> Tensor<1>;
}

#[derive(Module, Debug)]
pub struct LinearVHead {
    layer: Linear
}

impl LinearVHead {
    pub fn new(d_input: usize, d_output: usize, device: &Device) -> Self {
        Self {
            layer: LinearConfig::new(d_input, d_output).init(device)
        }
    }
}

impl VHead for LinearVHead {
    fn forward(&self, encoded: Tensor<2>) -> Tensor<1> {
        let x = self.layer.forward(encoded);
        x.squeeze_dim(1)
    }
}