//! configuration for loss functions of `burn`
//! - used in computing the loss of deep reinforcement learning algorithms

use burn::prelude::*;
/// total loss
pub trait Loss {
    /// returns the loss
    fn loss(&self) -> Tensor<1>;
}

/// additional loss trait for actor-critic methods
/// - Mainly for encoder-separated actor-critic method
pub trait ActorCriticLoss: Loss {
    /// return the policy loss, containing the entropy
    fn policy_loss_with_entropy(&self) -> Tensor<1>;
    /// return the value loss
    fn value_loss(&self) -> Tensor<1>;
}

/// A loss enumeration
#[derive(Debug, Clone, Copy)]
pub enum LossFn {
    /// Enumeration for mean squared loss
    MseLoss,
    /// Enumeration for Huber Loss
    HuberLoss {
        /// The bound where the Huber loss function changes from quadratic to linear behaviour.
        delta: f32
    },
    /// Enumeration for smooth l1 loss
    SmoothL1Loss {
        /// Specifies the threshold at which to change between L1 and L2 loss.
        beta: f32
    }
}

impl LossFn {
    /// compute the loss with mean reduction
    pub fn forward<const D: usize>(&self, predictions: Tensor<D>, targets: Tensor<D>) -> Tensor<1> {
        match self {
            LossFn::MseLoss => { burn::nn::loss::MseLoss::new().forward_no_reduction(predictions, targets).mean() },
            LossFn::HuberLoss { delta } => { burn::nn::loss::HuberLossConfig::new(*delta).init().forward_no_reduction(predictions, targets).mean() },
            LossFn::SmoothL1Loss { beta } => { burn::nn::loss::SmoothL1LossConfig::new().with_beta(*beta).init().forward(predictions, targets).mean() }
        }
    }

    /// compute the loss without reduction
    pub fn forward_no_reduction<const D: usize>(&self, predictions: Tensor<D>, targets: Tensor<D>) -> Tensor<D> {
        match self {
            LossFn::MseLoss => { burn::nn::loss::MseLoss::new().forward_no_reduction(predictions, targets) },
            LossFn::HuberLoss { delta } => { burn::nn::loss::HuberLossConfig::new(*delta).init().forward_no_reduction(predictions, targets) },
            LossFn::SmoothL1Loss { beta } => { burn::nn::loss::SmoothL1LossConfig::new().with_beta(*beta).init().forward(predictions, targets) }
        }
    }
}