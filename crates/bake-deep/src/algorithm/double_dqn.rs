use burn::{Tensor, tensor::Int};

use crate::{algorithm::dqn::ValueLoss, approximator::QFunction, constraint::DiscreteConstraint, types::{Batch, Batchable}};

pub struct DoubleDqn {
    pub gamma: f32,
    pub value_loss: ValueLoss,
}

pub struct DoubleDqnLoss {
    pub loss: Tensor<1>,
    pub td_error: Tensor<1>,
    pub qmean: Tensor<1>
}

impl DoubleDqn {
    pub fn new(gamma: f32, value_loss: ValueLoss) -> Self {
        Self { gamma, value_loss }
    }

    pub fn loss<Q, Obs, Constraint>(config: &DoubleDqn, online: &Q, target: &Q, batch: Batch<Obs, Tensor<1, Int>, Constraint>) -> DoubleDqnLoss
    where
        Q: QFunction<Obs = Obs>,
        Obs: Batchable,
        Constraint: DiscreteConstraint
    {
        let qvalues = online.forward(batch.obss, batch.constraints);
        let qvalues = qvalues.gather(1, batch.actions.unsqueeze_dim(1)).squeeze_dim::<1>(1);

        let next_qvalues_online = online.forward(batch.next_obss.clone(), batch.next_constraints.clone()).detach();
        let argmax = next_qvalues_online.argmax(1);
        let next_qvalues_target = target.forward(batch.next_obss, batch.next_constraints).detach();
        let next_qvalues = next_qvalues_target.gather(1, argmax).squeeze_dim::<1>(1);
        let targets = (batch.rewards + config.gamma * next_qvalues * (1f32 - batch.terminated)).detach();

        let td_error = targets.clone() - qvalues.clone();
        let qmean = qvalues.clone().mean();
        let loss = config.value_loss.forward(qvalues, targets);

        DoubleDqnLoss { loss, td_error, qmean }
    }
}