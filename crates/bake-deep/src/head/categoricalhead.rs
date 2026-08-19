//! Categorical Head for policy-based methods
use std::marker::PhantomData;

use burn::{Tensor, module::Module, nn::{Linear, LinearConfig}, tensor::Device};

use crate::{distribution::Categorical, head::Head, types::DiscreteConstraint};

#[derive(Module, Debug)]
pub struct CategoricalHead<Constraint> {
    layer: Linear,

    #[module(skip)]
    _c: PhantomData<Constraint>
}

impl<Constraint: DiscreteConstraint> CategoricalHead<Constraint> {
    pub fn new(d_input: usize, d_output: usize, device: &Device) -> Self {
        Self {
            layer: LinearConfig::new(d_input, d_output).init(device),
            _c: PhantomData,
        }
    }
}

impl<Constraint: DiscreteConstraint> Head for CategoricalHead<Constraint> {
    type Output = Categorical;
    type Constraint = Constraint;
    /// currently, fill_value is not used here
    fn forward(&self, encoded: Tensor<2>, constraint: Self::Constraint) -> Self::Output {
        let logits = self.layer.forward(encoded);
        Categorical::new(logits, constraint)
    }
}