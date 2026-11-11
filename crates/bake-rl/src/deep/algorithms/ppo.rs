//! Proximal Policy Optimization

use burn::prelude::*;
use crate::core::data::{Shaped, BatchOps, BatchedTransition, key};
use crate::deep::distribution::{Distribution, PossibleConstraint};
use crate::deep::algorithms::advantage_estimator::AdvantageEstimator;
use crate::deep::loss::{Loss, LossFn, ActorCriticLoss};
use crate::deep::contract::ActorCritic;

/// state struct for Proximal Policy Optimization algorithm
#[derive(Debug, Clone)]
pub struct PPOState {
    /// discount rate
    pub gamma: f32,
    /// entropy coefficient
    pub c_e: f32,
    /// critic coefficient
    /// # Warning
    /// - The critic coefficient will not be used when the actor and critic does not share the encoder
    pub c_c: f32,
    /// clipping
    pub eps: f32,
    /// for advantage calculation
    pub advantage: AdvantageEstimator,
    /// loss function for critic
    pub loss_fn: LossFn,
}

/// loss struct for PPO
#[derive(Debug, Clone)]
pub struct PPOLoss {
    /// total loss
    pub loss: Tensor<1>,
    /// policy loss with entropy
    pub policy_loss_with_entropy: Tensor<1>,
    /// policy loss
    pub policy_loss: Tensor<1>,
    /// value loss
    pub value_loss: Tensor<1>,
    /// entropy (no autodiff)
    pub entropy: Tensor<1>,
    /// approximate KL divergence (no autodiff)
    pub approx_kl: Tensor<1>,
    /// clipped amount (no autodiff)
    pub clip_fraction: Tensor<1>,
}

impl Loss for PPOLoss {
    fn loss(&self) -> Tensor<1> {
        self.loss.clone()
    }
}

impl ActorCriticLoss for PPOLoss {
    fn policy_loss_with_entropy(&self) -> Tensor<1> {
        self.policy_loss_with_entropy.clone()
    }

    fn value_loss(&self) -> Tensor<1> {
        self.value_loss.clone()
    }
}

/// compute the loss of PPO
/// # Warning
/// - Here, the given ActorCritic is moved to autodiff device
/// - Then ActorCritic will be moved to inner device when `update` is called
pub fn ppo_loss<Ac: ActorCritic>(
    state: &PPOState, actor_critic: Ac,
    minibatch: Shaped<BatchedTransition<Ac::Obs, impl PossibleConstraint<Ac::Dist>, <Ac::Dist as Distribution>::Sample, Tensor<1>>>
) -> (Ac, PPOLoss) {
    let ac = actor_critic.train();
    let (mut mb, _) = minibatch.autodiff().destruct();

    // compute log ratio, adv, ret
    let (dist, values) = ac.dist_and_state_value(mb.obs, mb.constraint);
    let log_ratio = dist.log_probs(mb.action) - mb.dict.remove(&key::log_prob).expect("`ppo_loss` expects key `log_prob`");
    let ratio = log_ratio.clone().exp();
    let adv = mb.dict.remove(&key::adv).expect("`ppo_loss` expects key `adv`");
    let ret = mb.dict.remove(&key::ret).expect("`ppo_loss` expects key `ret");

    // compute losses
    let stacked =
        Tensor::stack::<2>(
            vec![
                ratio.clone() * adv.clone(),
                ratio.clone().clamp(1f32 - state.eps, 1f32 + state.eps) * adv
                ], 1
            );
    let policy_loss = -stacked.min_dim(1).mean();
    let value_loss = state.loss_fn.forward(values.clone(), ret);
    let entropy = dist.entropy().mean();

    let policy_loss_with_entropy = policy_loss.clone() - state.c_e * entropy.clone();
    let loss = policy_loss_with_entropy.clone() + state.c_c * value_loss.clone();
    
    // compute logging metrics
    let approx_kl = ((log_ratio.clone().exp() - 1f32) - log_ratio.clone()).mean().detach().inner();
    let clip_fraction = (log_ratio.exp() - 1f32).abs().greater_elem(state.eps).float().mean().detach().inner();

    (ac, PPOLoss { loss, policy_loss, policy_loss_with_entropy, value_loss, entropy: entropy.inner(), approx_kl, clip_fraction })
}