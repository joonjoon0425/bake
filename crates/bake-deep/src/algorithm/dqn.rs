//! A Deep-QNetwork algorithm
use burn::prelude::*;
use crate::algorithm::loss_enum::Loss;

/// state for DQN
#[derive(Debug)]
pub struct Dqn {
    /// discount factor
    pub gamma: f32,
    /// loss function
    pub loss_fn: Loss,
}

/// A loss struct for Dqn
#[derive(Debug, Clone)]
pub struct DqnLoss {
    /// loss
    pub loss: Tensor<1>,
    /// temporal-difference error
    pub td_error: Tensor<1>,
    /// q-value mean
    pub qmean: Tensor<1>,
}

impl Dqn {
    
}