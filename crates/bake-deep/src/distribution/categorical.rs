//! Categorial distribution implementation
use burn::{Tensor, tensor::{Int, activation::log_softmax}};
use crate::{constraint::{Unconstrained, discrete_constraint::{DiscreteConstraint, DiscreteMask}}, distribution::{Distribution, PossibleConstraint}};

#[derive(Debug, Clone)]
/// A Categorical distribution
pub struct Categorical {
    log_probs: Tensor<2>,
}

impl Categorical {
    /// create a new Categorical distrinution
    pub fn new(params: <Self as Distribution>::Params, constraint: impl DiscreteConstraint) -> Self {
        let logits = constraint.apply(params, -1e9);
        let log_probs = log_softmax(logits.clone(), 1);
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
        self.log_probs.clone().gather(1, action.unsqueeze_dim(1)).squeeze_dim(1)
    }
    fn entropy(&self) -> Tensor<1> {
        let probs = self.log_probs.clone().exp();
        -(probs * self.log_probs.clone()).sum_dim(1).squeeze_dim(1)
    }
}

impl PossibleConstraint<Categorical> for DiscreteMask {
    fn create_distribution(params: <Categorical as Distribution>::Params, constraint: Self) -> Categorical {
        Categorical::new(params, constraint)
    }
}

impl PossibleConstraint<Categorical> for Unconstrained {
    fn create_distribution(params: <Categorical as Distribution>::Params, constraint: Self) -> Categorical {
        Categorical::new(params, constraint)
    }
}