//! An enumeration for advantage computation of Actor-Critic methods
//! 
use burn::prelude::*;

use crate::{contract::ActorCritic, data::{Batch, Batchable}, distribution::{Distribution, PossibleConstraint}};

/// Advantage computation enumeration
#[derive(Debug, Clone)]
pub enum AdvantageEstimator {
    /// 1-step TD residual
    Td0,
    /// Monte Carlo residual
    Td1 { 
        /// the number of environments
        n_envs: usize
    },
    /// generalized advatage estimation
    Gae {
        /// controls bias-variance tradeoff
        /// 0 -> low variance, high bias
        /// 1 -> high variance, low bias
        lambda: f32,
        /// the number of environments
        n_envs: usize
    }
}

impl AdvantageEstimator {
    /// compute the advantage
    pub fn advantage<Ac: ActorCritic>(
        &self,
        actor_critic: &Ac,
        batch: Batch<Ac::Obs, <Ac::Dist as Distribution>::Sample, impl PossibleConstraint<Ac::Dist>>,
        gamma: f32) -> (Tensor<1>, Tensor<1>)
    {
        match self {
            AdvantageEstimator::Td0 => AdvantageEstimator::td0(actor_critic, batch, gamma),
            AdvantageEstimator::Td1 { n_envs } => AdvantageEstimator::td1(actor_critic, batch, gamma, *n_envs),
            AdvantageEstimator::Gae { lambda, n_envs } => AdvantageEstimator::gae(actor_critic, batch, gamma, *lambda, *n_envs)
        }
    }

    fn td0<Ac: ActorCritic>(
        actor_critic: &Ac,
        batch: Batch<Ac::Obs, <Ac::Dist as Distribution>::Sample, impl PossibleConstraint<Ac::Dist>>,
        gamma: f32
    ) -> (Tensor<1>, Tensor<1>) {
        let values = actor_critic.value(batch.obss);
        let next_values = actor_critic.value(batch.next_obss);
        let adv = batch.rewards + gamma * next_values * (1f32 - batch.terminated.clone()) - values.clone();
        
        let returns = adv.clone() + values;
        (adv.detach(), returns.detach())
    }

    fn td1<Ac: ActorCritic>(
        actor_critic: &Ac,
        batch: Batch<Ac::Obs, <Ac::Dist as Distribution>::Sample, impl PossibleConstraint<Ac::Dist>>,
        gamma: f32, n_envs: usize,
    ) -> (Tensor<1>, Tensor<1>) {
        let n = batch.batch_size().unwrap();
        let device = batch.device();
        let values = actor_critic.value(batch.obss);
        let next_values = actor_critic.value(batch.next_obss);
        let deltas = batch.rewards + gamma * next_values * (1f32 - batch.terminated.clone()) - values.clone();
        let terminated = batch.terminated;
        let truncated = batch.truncated;
        
        let mut adv = Tensor::zeros([n], &device).detach();
        adv = adv.slice_assign(n - n_envs..n, deltas.clone().slice(n - n_envs..n));
        for i in (0..n - n_envs).step_by(n_envs).rev() {
            adv = adv.clone().slice_assign(i..i + n_envs, deltas.clone().slice(i..i + n_envs) + gamma * adv.clone().slice(i + n_envs..i + 2 * n_envs) * (1f32 - truncated.clone().slice(i..i + n_envs)) * (1f32 - terminated.clone().slice(i..i + n_envs)));
        }
        let returns = adv.clone() + values;
        (adv.detach(), returns.detach())
    }

    fn gae<Ac: ActorCritic>(
        actor_critic: &Ac,
        batch: Batch<Ac::Obs, <Ac::Dist as Distribution>::Sample, impl PossibleConstraint<Ac::Dist>>,
        gamma: f32,
        lambda: f32, n_envs: usize
    ) -> (Tensor<1>, Tensor<1>) {
        let n = batch.batch_size().unwrap();
        let device = batch.device();
        let values = actor_critic.value(batch.obss);
        let next_values = actor_critic.value(batch.next_obss);
        let deltas = batch.rewards + gamma * next_values * (1f32 - batch.terminated.clone()) - values.clone();
        let terminated = batch.terminated;
        let truncated = batch.truncated;
        
        let mut adv = Tensor::zeros([n], &device).detach();
        adv = adv.slice_assign(n - n_envs..n, deltas.clone().slice(n - n_envs..n));
        for i in (0..n - n_envs).step_by(n_envs).rev() {
            adv = adv.clone().slice_assign(i..i + n_envs, deltas.clone().slice(i..i + n_envs) + gamma * lambda * adv.clone().slice(i + n_envs..i + 2 * n_envs) * (1f32 - truncated.clone().slice(i..i + n_envs)) * (1f32 - terminated.clone().slice(i..i + n_envs)));
        }
        let returns = adv.clone() + values;
        (adv.detach(), returns.detach())
    }
}