//! A Double Deep-QNetwork algorithm implementation
use burn::{optim::{GradientsParams, ModuleOptimizer}, prelude::*};
use crate::{algorithm::loss_enum::Loss, buffer::sampler::SampleInfo, constraint::discrete_constraint::DiscreteConstraint, contract::DiscreteQFunction, data::Batch, logger::ToLog};

/// state for Double DQN
#[derive(Debug, Clone)]
pub struct DoubleDqn {
    /// discount factor
    pub gamma: f32,
    /// loss function
    pub loss_fn: Loss,
}

/// A loss struct for DoubleDqn
#[derive(Debug, Clone)]
pub struct DoubleDqnLoss {
    /// loss
    pub loss: Tensor<1>,
    /// temporal-difference error
    pub td_error: Tensor<1>,
    /// q-value mean
    pub qmean: Tensor<1>,
}

impl DoubleDqn {
    /// compute the loss for Dqn algorithm. If `is_weight` is not `None` in `batch_info`, the weighted loss is returned,
    pub fn loss<Q, Constraint>(state: &DoubleDqn, online: &Q, target: &Q, batch: Batch<Q::Obs, Tensor<1, Int>, Constraint>, batch_info: SampleInfo) -> DoubleDqnLoss
    where
        Q: DiscreteQFunction,
        Constraint: DiscreteConstraint
    {
        let qvalues = online.forward(batch.obss, batch.constraints);
        let qvalues = qvalues.gather(1, batch.actions.unsqueeze_dim(1)).squeeze_dim::<1>(1);

        let next_qvalues_online = target.forward(batch.next_obss.clone(), batch.next_constraints.clone()).detach();
        let argmax = next_qvalues_online.argmax(1);
        let next_qvalues_target = target.forward(batch.next_obss, batch.next_constraints).detach();
        let next_qvalues = next_qvalues_target.gather(1, argmax).squeeze_dim::<1>(1);
        let targets = ((batch.rewards + state.gamma * next_qvalues.max_dim(1).squeeze_dim(1)) * (1f32 - batch.terminated)).detach();

        let td_error = (targets.clone() - qvalues.clone()).detach();
        let qmean = qvalues.clone().detach().mean();

        match batch_info.is_weights {
            Some(is_weights) => {
                let loss = (state.loss_fn.forward_no_reduction(qvalues, targets) * is_weights).mean();
                DoubleDqnLoss { loss, td_error, qmean }
            },
            None => {
                let loss = state.loss_fn.forward(qvalues, targets);
                DoubleDqnLoss { loss, td_error, qmean }
            }
        }
    }

    /// update the Q function with given learning rate and optimizer
    pub fn update<Q: DiscreteQFunction>(online: Q, loss: DoubleDqnLoss, lr: f64, opt: &mut ModuleOptimizer) -> Q {
        let grads = loss.loss.backward();
        let grads = GradientsParams::from_grads(grads, &online);
        opt.step(lr, online, grads)
    }

    /// gives the name of recordable logs. use it to register at the logger
    pub fn log_names() -> Vec<&'static str> {
        vec![
            "loss",
            "mean_td_error",
            "qmean"
        ]
    }
}

impl ToLog for DoubleDqnLoss {
    fn to_log(&self) -> std::collections::HashMap<&'static str, f32> {
        let mut log = std::collections::HashMap::new();
        log.insert("loss", self.loss.clone().into_scalar());
        log.insert("mean_td_error", self.td_error.clone().mean().into_scalar());
        log.insert("qmean", self.qmean.clone().into_scalar());

        log
    }
}