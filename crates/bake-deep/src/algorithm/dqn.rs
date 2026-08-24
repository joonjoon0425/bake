//! A configuration for Dqn algorithm

use std::collections::HashMap;

use burn::{Tensor, config::Config, nn::loss::{HuberLossConfig, MseLoss, Reduction}, optim::{GradientsParams, ModuleOptimizer}, tensor::Int};
use crate::{approximator::QFunction, constraint::DiscreteConstraint, types::{Batch, Batchable, Recordable}};

#[derive(Debug, Config)]
pub struct Dqn {
    pub gamma: f32,
    pub value_loss: ValueLoss,
}

#[derive(Debug, Clone)]
pub struct DqnLoss {
    pub loss: Tensor<1>,
    pub td_error: Tensor<1>,
    pub qmean: Tensor<1>
}

impl Dqn {
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

        let td_error = (targets.clone() - qvalues.clone()).detach();
        let qmean = qvalues.clone().detach().mean();
        let loss = config.value_loss.forward(qvalues, targets);

        DqnLoss { loss, td_error, qmean, }
    }

    pub fn update<Q: QFunction>(online: Q, loss: DqnLoss, lr: f64, opt: &mut ModuleOptimizer) -> Q {
        let grads = loss.loss.backward();
        let grads = GradientsParams::from_grads(grads, &online);
        opt.step(lr, online, grads)
    }
}

impl Recordable for DqnLoss {
    fn to_record(&self) -> HashMap<&'static str, Tensor<1>> {
        let mut record = HashMap::new();
        record.insert("loss", self.loss.clone().detach());
        record.insert("td_error", self.td_error.clone().detach());
        record.insert("qmean", self.qmean.clone().detach());
        record
    }
}

#[derive(Debug, Config)]
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