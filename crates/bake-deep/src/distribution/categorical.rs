//! Categorial distribution implementation
use burn::{Tensor, tensor::{Int, activation::log_softmax}};
use crate::{constraint::DiscreteConstraint, distribution::Distribution};

pub struct Categorical {
    logits: Tensor<2>,
    log_probs: Tensor<2>,
}

impl Categorical {
    pub fn new(logits: Tensor<2>, constraint: impl DiscreteConstraint) -> Self {
        let logits = constraint.apply(logits, -1e9);
        let log_probs = log_softmax(logits.clone(), 1);
        Self {
            logits,
            log_probs,
        }
    }

    pub fn logits(&self) -> &Tensor<2> { &self.logits }
}

impl Distribution for Categorical {
    type Action = Tensor<1, Int>;
    
    fn sample(&self) -> Self::Action {
        self.log_probs.clone().exp().categorical(1).squeeze_dim(1)
    }
    fn mode(&self) -> Self::Action {
        self.log_probs.clone().argmax(1).squeeze_dim(1)
    }
    fn log_probs(&self, action: Self::Action) -> Tensor<1> {
        self.log_probs.clone().gather(1, action.unsqueeze_dim(1)).squeeze_dim(1)
    }
    fn entropy(&self) -> Tensor<1> {
        let probs = self.log_probs.clone().exp();
        -(probs * self.log_probs.clone()).sum_dim(1).squeeze_dim(1)
    }
}