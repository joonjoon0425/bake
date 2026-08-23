//! A configuration for Dqn algorithm

use burn::{Tensor, nn::loss::{HuberLossConfig, MseLoss, Reduction}, tensor::Int};
use crate::{approximator::QFunction, constraint::DiscreteConstraint, types::{Batch, Batchable}};

pub struct Dqn {
    pub gamma: f32,
    pub value_loss: ValueLoss,
}

pub struct DqnLoss {
    pub loss: Tensor<1>,
    pub td_error: Tensor<1>,
    pub qmean: Tensor<1>
}

impl Dqn {
    pub fn new(gamma: f32, value_loss: ValueLoss) -> Self {
        Self { gamma, value_loss }
    }

    pub fn loss<Q, Obs, Constraint>(config: &Dqn, online: &Q, target: &Q, batch: Batch<Obs, Tensor<1, Int>, Constraint>) -> DqnLoss
    where
        Q: QFunction<Obs = Obs>,
        Obs: Batchable,
        Constraint: DiscreteConstraint
    {
        let qvalues = online.forward(batch.obss, batch.constraints);
        let qvalues = qvalues.gather(1, batch.actions.unsqueeze_dim(1)).squeeze_dim::<1>(1);

        let next_qvalues = target.forward(batch.next_obss, batch.next_constraints).detach();
        let targets = (batch.rewards + config.gamma * next_qvalues.max_dim(1).squeeze_dim(1) * (1f32 - batch.terminated)).detach();

        let td_error = targets.clone() - qvalues.clone();
        let qmean = qvalues.clone().mean();
        let loss = config.value_loss.forward(qvalues, targets);

        DqnLoss { loss, td_error, qmean, }
    }
}

pub enum ValueLoss {
    MseLoss,
    HuberLoss{ delta: f32 },
}

impl ValueLoss {
    pub fn forward<const D: usize>(&self, logits: Tensor<D>, targets: Tensor<D>) -> Tensor<1> {
        match self {
            ValueLoss::MseLoss => {
                MseLoss::new().forward(logits, targets, Reduction::Mean)
            },
            ValueLoss::HuberLoss { delta } => {
                HuberLossConfig::new(*delta).init().forward(logits, targets, Reduction::Mean)
            }
        }
    }
}