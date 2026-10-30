//! Proximal Policy Optimization algorithm
use burn::prelude::*;
use crate::core::data::{Batch, BatchTensor, DetachBatchDim, WithBatchDim};
use crate::core::distribution::{Distribution, PossibleConstraint};
use crate::deep::algorithms::advantage_estimator::AdvantageEstimator;
use crate::deep::contract::ActorCritic;
use crate::deep::{
    loss::{Loss, LossFn}
};

/// Proximal Policy Optimization algorithm state
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

/// Proximal Policy Optimization loss
#[derive(Debug, Clone)]
pub struct PPOLoss {
    /// total loss
    pub total_loss: Tensor<1>,
    /// actor loss with entropy
    pub policy_loss_with_entropy: Tensor<1>,
    /// loss of policy (surrogate)
    pub policy_loss: Tensor<1>,
    /// loss of critic (value)
    pub value_loss: Tensor<1>,
    /// entropy (how probabilistic is the policy)
    pub entropy: Tensor<1>,

    /// approximate KL divergence
    pub approx_kl: Tensor<1>,
    /// clipped ratio
    pub clip_fraction: Tensor<1>,
}

pub fn ppo_loss<Ac: ActorCritic>(state: &PPOState, ac: Ac, minibatch: Batch<Ac::Obs, <Ac::Dist as Distribution>::Sample, impl PossibleConstraint<Ac::Dist>, BatchTensor<1>>) -> (Ac, PPOLoss) {
    let actor_critic = ac.train();
        let mut minibatch = minibatch.autodiff();

        let (dist, values) = actor_critic.dist_and_state_value(minibatch.obs, minibatch.constraint);
        let log_ratio = dist.log_probs(minibatch.action) - minibatch.extras.remove::<LogProb>().unwrap();
        let ratio = log_ratio.clone().exp();
        let adv = minibatch.extras.remove::<Advantage>().unwrap();
        let ret = minibatch.extras.remove::<Return>().unwrap();

        let stacked = Tensor::stack::<2>(vec![ratio.clone() * adv.clone(), ratio.clone().clamp(1f32 - state.eps, 1f32 + state.eps) * adv], 1);
        let policy_loss = -stacked.min_dim(1).mean();
        let value_loss = state.loss_fn.forward(values.clone(), ret);
        let entropy = dist.entropy().mean();
        
        let policy_loss_with_entropy = policy_loss.clone() - state.c_e * entropy.clone();
        let total_loss = policy_loss_with_entropy.clone() + state.c_c * value_loss.clone();

        let approx_kl = ((log_ratio.clone().exp() - 1f32) - log_ratio.clone()).mean().detach();
        let clip_ratio = (log_ratio.exp() - 1f32).abs().greater_elem(state.eps).float().mean().detach();

        (actor_critic, PpoLoss { total_loss, policy_loss_with_entropy, policy_loss, value_loss, entropy, approx_kl, clip_fraction: clip_ratio })
    
}