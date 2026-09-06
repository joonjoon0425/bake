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
    Td1,
    /// generalized advatage estimation
    Gae {
        /// controls bias-variance tradeoff
        /// 0 -> low variance, high bias
        /// 1 -> high variance, low bias
        lambda: f32
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
            AdvantageEstimator::Td1 => AdvantageEstimator::td1(actor_critic, batch, gamma),
            AdvantageEstimator::Gae { lambda } => AdvantageEstimator::gae(actor_critic, batch, gamma, *lambda)
        }
    }

    fn td0<Ac: ActorCritic>(
        actor_critic: &Ac,
        batch: Batch<Ac::Obs, <Ac::Dist as Distribution>::Sample, impl PossibleConstraint<Ac::Dist>>,
        gamma: f32,
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
        gamma: f32,
    ) -> (Tensor<1>, Tensor<1>) {
        let n = batch.len().unwrap();
        let device = batch.device();
        let values = actor_critic.value(batch.obss);
        let next_values = actor_critic.value(batch.next_obss);
        let deltas = batch.rewards + gamma * next_values * (1f32 - batch.terminated.clone()) - values.clone();
        let terminated = batch.terminated;
        let truncated = batch.truncated;
        
        let mut adv = Tensor::zeros([n], &device).detach();
        adv = adv.slice_assign(n - 1..n, deltas.clone().slice(n - 1..n));
        for i in (0..n-1).rev() {
            adv = adv.clone().slice_assign(i..i + 1, deltas.clone().slice(i..i + 1) + gamma * adv.clone().slice(i + 1..i + 2) * (1f32 - truncated.clone().slice(i..i + 1)) * (1f32 - terminated.clone().slice(i..i + 1)));
        }
        let returns = adv.clone() + values;
        (adv.detach(), returns.detach())
    }

    fn gae<Ac: ActorCritic>(
        actor_critic: &Ac,
        batch: Batch<Ac::Obs, <Ac::Dist as Distribution>::Sample, impl PossibleConstraint<Ac::Dist>>,
        gamma: f32,
        lambda: f32
    ) -> (Tensor<1>, Tensor<1>) {
        let n = batch.len().unwrap();
        let device = batch.device();
        let values = actor_critic.value(batch.obss);
        let next_values = actor_critic.value(batch.next_obss);
        let deltas = batch.rewards + gamma * next_values * (1f32 - batch.terminated.clone()) - values.clone();
        let terminated = batch.terminated;
        let truncated = batch.truncated;
        
        let mut adv = Tensor::zeros([n], &device).detach();
        adv = adv.slice_assign(n - 1..n, deltas.clone().slice(n - 1..n));
        for i in (0..n-1).rev() {
            adv = adv.clone().slice_assign(i..i + 1, deltas.clone().slice(i..i + 1) + gamma * lambda * adv.clone().slice(i + 1..i + 2) * (1f32 - truncated.clone().slice(i..i + 1)) * (1f32 - terminated.clone().slice(i..i + 1)));
        }
        let returns = adv.clone() + values;
        (adv.detach(), returns.detach())
    }
}