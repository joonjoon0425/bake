//! Categorical distribution

use burn::tensor::activation::log_softmax;
use burn::prelude::*;
use crate::core::data::batch_dim::convert::{AttachBatchDim, DetachBatchDim};
use crate::core::data::batch_dim::*;
use crate::core::{distribution::{Distribution, PossibleConstraint}};
use crate::core::constraint::DiscreteConstraint;

#[derive(Debug, Clone)]
/// A Categorical probability distribution
pub struct Categorical {
    log_probs: BatchTensor<2>,
}

impl Categorical {
    /// create a new `Categorical` distribution
    pub fn new(params: <Self as Distribution>::Params, constraint: impl DiscreteConstraint) -> Self {
        let (logits, bs) = constraint.apply(params, -1e9).detach_batch_dim();
        let log_probs = log_softmax(logits, 1).attach_batch_dim(bs);
        Self {
            log_probs
        }
    }
}

impl Distribution for Categorical {
    type Sample = BatchTensor<1, Int>;
    type Params = BatchTensor<2>;

    fn sample(&self) -> Self::Sample {
        let (log_probs, bs) = self.log_probs.clone().detach_batch_dim();
        log_probs.exp().categorical(1).squeeze_dim(1).attach_batch_dim(bs)
    }

    fn mode(&self) -> Self::Sample {
        let (log_probs, bs) = self.log_probs.clone().detach_batch_dim();
        log_probs.argmax(1).squeeze_dim(1).attach_batch_dim(bs)
    }

    fn log_probs(&self, action: Self::Sample) -> BatchTensor<1> {
        let (log_probs, bs) = self.log_probs.clone().detach_batch_dim();
        let (action, abs) = action.detach_batch_dim();
        assert!(bs == abs, "Categorical::log_probs: batch dimension does not matches");
        log_probs.gather(1, action.unsqueeze_dim(1)).squeeze_dim(1).attach_batch_dim(bs)
    }
    
    fn entropy(&self) -> BatchTensor<1> {
        let (log_probs, bs) = self.log_probs.clone().detach_batch_dim();
        let probs = log_probs.clone().exp();
        (-probs * log_probs.clone()).sum_dim(1).squeeze_dim(1).attach_batch_dim(bs)
    }
}

/// blanket impl for Categorical distribution
impl<T: DiscreteConstraint> PossibleConstraint<Categorical> for T {
    fn create_distribution(params: <Categorical as Distribution>::Params, constraint: Self) -> Categorical {
        Categorical::new(params, constraint)
    }
}