//! An Advantage Actor-Critic algorithm implementation
use std::collections::HashMap;

use bake_common::logger::ToLog;
use burn::prelude::*;
use crate::{algorithm::advantage_estimator::AdvantageEstimator, contract::compound::ActorCritic, data::{Batch, Batchable}, distribution::{Distribution, PossibleConstraint}, loss::Loss};

/// state for A2C
#[derive(Debug, Clone)]
pub struct A2C {
    /// discount rate
    pub gamma: f32,
    /// for advantage calculation
    pub advantage: AdvantageEstimator,
    /// loss function for critic
    pub loss_fn: Loss,
}

/// loss struct for A2C
#[derive(Debug, Clone)]
pub struct A2CLoss {
    /// loss of policy (actor)
    pub actor_loss: Tensor<1>,
    /// loss of value (critic)
    pub critic_loss: Tensor<1>,
    /// entropy
    pub entropy: Tensor<1>,
}

impl A2C {
    /// compute the loss of A2C
    /// # Warning
    /// - Here, the given ActorCritic is moved to autodiff device
    /// - The ActorCritic will be move to inner device when `update` is called
    pub fn loss<Ac: ActorCritic>(state: &A2C, actor_critic: Ac, batch: Batch<Ac::Obs, <Ac::Dist as Distribution>::Sample, impl PossibleConstraint<Ac::Dist>>) -> (Ac, A2CLoss) {
        let actor_critic = actor_critic.train();
        let batch = batch.into_autodiff();
        let (dist, values) = actor_critic.dist_and_state_value(batch.obss.clone(), batch.constraints.clone());
        let (adv, ret) = state.advantage.advantage(&actor_critic, batch.clone(), state.gamma);
        // 1. advantage
        let adv = (adv.clone() - adv.clone().mean()) / (adv.clone().var(0) + 1e-9).sqrt();
        // 2. policy surrogate
        let log_prob = dist.log_probs(batch.actions);
        let actor_loss = -(log_prob * adv).mean();
        // 3. entropy
        let entropy = dist.entropy().mean();
        // 4. value loss
        let critic_loss = state.loss_fn.forward(values, ret);

        (actor_critic, A2CLoss { actor_loss, critic_loss, entropy })
    }

    /// gives the name of recordable logs. use it to register at the logger
    pub fn log_names() -> Vec<&'static str> {
        vec![
            "actor_loss",
            "critic_loss",
            "entropy"
        ]
    }
}

impl ToLog for A2CLoss {
    fn to_log(&self) -> HashMap<&'static str, f32> {
        let mut record = HashMap::new();
        record.insert("actor_loss", self.actor_loss.clone().into_scalar());
        record.insert("critic_loss", self.critic_loss.clone().into_scalar());
        record.insert("entropy", self.entropy.clone().into_scalar());
        record
    }
}
