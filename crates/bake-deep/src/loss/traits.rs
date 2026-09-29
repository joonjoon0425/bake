//! A trait for algorithm losses
//! - Created for `update` functions
//! 
use burn::prelude::*;

/// total loss
pub trait TotalLoss {
    /// returns the total loss
    fn total_loss(&self) -> Tensor<1>;
}

/// additional loss trait for actor-critic methods
/// - Mainly for encoder-separated actor-critic method
pub trait ActorCriticLoss: TotalLoss {
    /// return the policy loss, containing the entropy
    fn policy_loss_with_entropy(&self) -> Tensor<1>;
    /// return the value loss
    fn value_loss(&self) -> Tensor<1>;
}