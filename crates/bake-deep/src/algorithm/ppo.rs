//! Proximal Policy Optimization algorithm implementation
//! 
use std::collections::HashMap;

use bake_common::logger::ToLog;
use burn::prelude::*;
use crate::{
    algorithm::advantage_estimator::AdvantageEstimator,
    contract::compound::ActorCritic,
    data::{Batch, Batchable, extras::*},
    distribution::{Distribution, PossibleConstraint}, 
    loss::Loss
};

/// state struct for PPO
pub struct Ppo {
    /// discount rate
    pub gamma: f32,
    /// clipping
    pub eps: f32,
    /// for advantage calculation
    pub advantage: AdvantageEstimator,
    /// loss function for critic
    pub loss_fn: Loss,
}

/// loss struct for PPO
pub struct PpoLoss {
    /// loss of policy (surrogate)
    pub actor_loss: Tensor<1>,
    /// loss of critic (value)
    pub critic_loss: Tensor<1>,
    /// entropy (how probabilistic is the policy)
    pub entropy: Tensor<1>,

    /// approximate KL divergence
    pub approx_kl: Tensor<1>,
    /// clipped ratio
    pub clip_fraction: Tensor<1>,
}

impl Ppo {
    /// compute the loss of PPO
    /// # Warning
    /// - Here, the given ActorCritic is moved to autodiff device
    /// - The ActorCritic will be move to inner device when `update` is called
    pub fn loss<Ac: ActorCritic>(state: &Ppo, actor_critic: Ac, minibatch: Batch<Ac::Obs, <Ac::Dist as Distribution>::Sample, impl PossibleConstraint<Ac::Dist>>) -> (Ac, PpoLoss) {
        let actor_critic = actor_critic.train();
        let mut minibatch = minibatch.into_autodiff();

        let (dist, values) = actor_critic.dist_and_state_value(minibatch.obss, minibatch.constraints);
        let log_ratio = dist.log_probs(minibatch.actions) - minibatch.extras.remove::<LogProb>().unwrap();
        let ratio = log_ratio.clone().exp();
        let adv = minibatch.extras.remove::<Advantage>().unwrap();
        let ret = minibatch.extras.remove::<Return>().unwrap();

        let stacked = Tensor::stack::<2>(vec![ratio.clone() * adv.clone(), ratio.clone().clamp(1f32 - state.eps, 1f32 + state.eps) * adv], 1);
        let actor_loss = -stacked.min_dim(1).mean();
        let critic_loss = state.loss_fn.forward(values.clone(), ret);
        let entropy = dist.entropy().mean();

        let approx_kl = ((log_ratio.clone().exp() - 1f32) - log_ratio.clone()).mean().detach();
        let clip_ratio = (log_ratio.exp() - 1f32).abs().greater_elem(state.eps).float().mean().detach();

        (actor_critic, PpoLoss { actor_loss, critic_loss, entropy, approx_kl, clip_fraction: clip_ratio })
    }

    /// gives the name of recordable logs. use it to register at the logger
    pub fn log_names() -> Vec<&'static str> {
        vec![
            "actor_loss",
            "critic_loss",
            "entropy",
            "approx_kl",
            "clip_fraction",
        ]
    }
}

impl ToLog for PpoLoss {
    fn to_log(&self) -> HashMap<&'static str, f32> {
        let mut record = HashMap::new();
        record.insert("actor_loss", self.actor_loss.clone().into_scalar());
        record.insert("critic_loss", self.critic_loss.clone().into_scalar());
        record.insert("entropy", self.entropy.clone().into_scalar());
        record.insert("approx_kl", self.approx_kl.clone().into_scalar());
        record.insert("clip_fraction", self.clip_fraction.clone().into_scalar());
        record
    }
}
