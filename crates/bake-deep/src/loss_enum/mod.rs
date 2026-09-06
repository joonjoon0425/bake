//! A loss enumeration which algorithm losses will use
//! Since burn do not have common trait for losses, I had to make it by myself
//! 
//! 
use burn::prelude::*;

/// A loss enumeration
#[derive(Debug, Clone, Copy)]
pub enum Loss {
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

impl Loss {
    /// compute the loss with mean reduction
    pub fn forward<const D: usize>(&self, predictions: Tensor<D>, targets: Tensor<D>) -> Tensor<1> {
        match self {
            Loss::MseLoss => { burn::nn::loss::MseLoss::new().forward_no_reduction(predictions, targets).mean() },
            Loss::HuberLoss { delta } => { burn::nn::loss::HuberLossConfig::new(*delta).init().forward_no_reduction(predictions, targets).mean() },
            Loss::SmoothL1Loss { beta } => { burn::nn::loss::SmoothL1LossConfig::new().with_beta(*beta).init().forward(predictions, targets).mean() }
        }
    }

    /// compute the loss without reduction
    pub fn forward_no_reduction<const D: usize>(&self, predictions: Tensor<D>, targets: Tensor<D>) -> Tensor<D> {
        match self {
            Loss::MseLoss => { burn::nn::loss::MseLoss::new().forward_no_reduction(predictions, targets) },
            Loss::HuberLoss { delta } => { burn::nn::loss::HuberLossConfig::new(*delta).init().forward_no_reduction(predictions, targets) },
            Loss::SmoothL1Loss { beta } => { burn::nn::loss::SmoothL1LossConfig::new().with_beta(*beta).init().forward(predictions, targets) }
        }
    }
}