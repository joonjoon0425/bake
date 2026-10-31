//! Categorical distribution

use burn::tensor::activation::log_softmax;
use burn::prelude::*;
use crate::core::{distribution::{Distribution, PossibleConstraint}};
use crate::core::constraint::DiscreteConstraint;

#[derive(Debug, Clone)]
/// A Categorical probability distribution
pub struct Categorical {
    log_probs: Tensor<2>,
}

impl Categorical {
    /// create a new `Categorical` distribution
    pub fn new(params: <Self as Distribution>::Params, constraint: impl DiscreteConstraint) -> Self {
        let logits = constraint.apply(params, -1e9);
        let log_probs = log_softmax(logits, 1);
        Self {
            log_probs
        }
    }
}

impl Distribution for Categorical {
    type Sample = Tensor<1, Int>;
    type Params = Tensor<2>;

    fn sample(&self) -> Self::Sample {
        self.log_probs.clone().exp().categorical(1).squeeze_dim(1)
    }

    fn mode(&self) -> Self::Sample {
        self.log_probs.clone().argmax(1).squeeze_dim(1)
    }

    fn log_probs(&self, action: Self::Sample) -> Tensor<1> {
        assert!(self.log_probs.dims()[0] == action.dims()[0], "Categorical::log_probs: batch dimension does not matches");
        self.log_probs.clone().gather(1, action.unsqueeze_dim(1)).squeeze_dim(1)
    }
    
    fn entropy(&self) -> Tensor<1> {
        let log_probs = self.log_probs.clone();
        let probs = log_probs.clone().exp();
        (-probs * log_probs).sum_dim(1).squeeze_dim(1)
    }
}

/// blanket impl for Categorical distribution
impl<T: DiscreteConstraint> PossibleConstraint<Categorical> for T {
    fn create_distribution(params: <Categorical as Distribution>::Params, constraint: Self) -> Categorical {
        Categorical::new(params, constraint)
    }
}