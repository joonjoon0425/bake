use burn::{Tensor, module::Module, nn::{Linear, LinearConfig}, tensor::Device};
use crate::{head::QHead, types::DiscreteMask};

#[derive(Module, Debug)]
pub struct LinearQHead {
    layer: Linear
}

impl LinearQHead {
    pub fn new(d_input: usize, d_output: usize, device: &Device) -> Self {
        Self {
            layer: LinearConfig::new(d_input, d_output).init(device)
        }
    }
}

impl QHead for LinearQHead {
    fn forward(&self, encoded: Tensor<2>, barrier: Option<DiscreteMask>) -> Tensor<2> {
        let x = self.layer.forward(encoded);
        match barrier {
            Some(mask) => mask.apply(x, -1e9),
            None => x
        }
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

impl QHead for DuelingQHead {
    fn forward(&self, encoded: Tensor<2>, barrier: Option<DiscreteMask>) -> Tensor<2> {
        let v = self.value.forward(encoded.clone());
        let a = self.advantage.forward(encoded);
        match barrier {
            Some(mask) => {
                let mean = mask.clone().mean_dim(1, a.clone());
                mask.apply(v + a - mean, -1e9)
            },
            None => {
                let mean = a.clone().mean_dim(1);
                v + a - mean
            }
        }
    }
}