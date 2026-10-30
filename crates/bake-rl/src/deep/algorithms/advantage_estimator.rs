//! An enumeration for advantage estimation of actor-critic metehods
use burn::prelude::*;

use crate::{core::{data::{AttachBatchDim, Batch, BatchTensor, DetachBatchDim, WithBatchDim, batch_dim::ops::n_batch}, distribution::{Distribution, PossibleConstraint}}, deep::contract::ActorCritic};

/// advantage estimator
#[derive(Debug, Clone)]
pub enum AdvantageEstimator {
    /// 1-step TD residual
    Td0,
    /// MC residual
    Td1,
    /// generalized advantage estimation
    Gae {
        /// controls bias-variance tradeoff
        /// 0 -> low var high bias
        /// 1 -> high var low bias
        lambda: f32,
    }
}

impl AdvantageEstimator {
    /// compute the advantage
    pub fn advantage<Ac: ActorCritic>(
        &self,
        actor_critic: &Ac,
        batch: Batch<Ac::Obs, <Ac::Dist as Distribution>::Sample, impl PossibleConstraint<Ac::Dist>, BatchTensor<1>>,
        gamma: f32
    ) -> (BatchTensor<1>, BatchTensor<1>)
    {
        match self {
            AdvantageEstimator::Td0 => AdvantageEstimator::td0(actor_critic, batch, gamma),
            AdvantageEstimator::Td1 => AdvantageEstimator::td1(actor_critic, batch, gamma),
            AdvantageEstimator::Gae { lambda } => AdvantageEstimator::gae(actor_critic, batch, gamma, *lambda)
        }
    }

    fn td0<Ac: ActorCritic>(
        actor_critic: &Ac,
        batch: Batch<Ac::Obs, <Ac::Dist as Distribution>::Sample, impl PossibleConstraint<Ac::Dist>, BatchTensor<1>>,
        gamma: f32
    ) -> (BatchTensor<1>, BatchTensor<1>) {
        let bs = batch.batch_shape();
        let values = actor_critic.state_value(batch.obs).detach_batch_dim().0;
        let next_values = actor_critic.state_value(batch.next_obs).detach_batch_dim().0;
        let adv = batch.reward.detach_batch_dim().0 + gamma * next_values * (1f32 - batch.terminated.clone().detach_batch_dim().0.float()) - values.clone();
        
        let returns = adv.clone() + values;
        (adv.detach().attach_batch_dim(bs.clone()), returns.detach().attach_batch_dim(bs))
    }

    fn td1<Ac: ActorCritic>(
        actor_critic: &Ac,
        batch: Batch<Ac::Obs, <Ac::Dist as Distribution>::Sample, impl PossibleConstraint<Ac::Dist>, BatchTensor<1>>,
        gamma: f32
    ) -> (BatchTensor<1>, BatchTensor<1>) {
        let bs = batch.batch_shape();
        let n_envs = bs[0];
        let n = n_batch(&bs);
        let device = batch.device();
        let values = actor_critic.state_value(batch.obs).detach_batch_dim().0;
        let next_values = actor_critic.state_value(batch.next_obs).detach_batch_dim().0;
        let deltas = batch.reward.detach_batch_dim().0 + gamma * next_values * (1f32 - batch.terminated.clone().detach_batch_dim().0.float()) - values.clone();
        let terminated = batch.terminated;
        let truncated = batch.truncated;
        
        let mut adv = Tensor::zeros([n], &device).detach();
        adv = adv.slice_assign(n - n_envs..n, deltas.clone().slice(n - n_envs..n));
        for i in (0..n - n_envs).step_by(n_envs).rev() {
            adv = adv.clone().slice_assign(i..i + n_envs, deltas.clone().slice(i..i + n_envs) + gamma * adv.clone().slice(i + n_envs..i + 2 * n_envs) * (1f32 - truncated.clone().slice(0, i..i + 1).detach_batch_dim().0.float()) * (1f32 - terminated.clone().slice(0, i..i + 1).detach_batch_dim().0.float()));
        }
        let returns = adv.clone() + values;
        (adv.detach().attach_batch_dim(bs.clone()), returns.detach().attach_batch_dim(bs))
    }

    fn gae<Ac: ActorCritic>(
        actor_critic: &Ac,
        batch: Batch<Ac::Obs, <Ac::Dist as Distribution>::Sample, impl PossibleConstraint<Ac::Dist>, BatchTensor<1>>,
        gamma: f32,
        lambda: f32
    ) -> (BatchTensor<1>, BatchTensor<1>) {
        let bs = batch.batch_shape();
        let n_envs = bs[0];
        let n = n_batch(&bs);
        let device = batch.device();
        let values = actor_critic.state_value(batch.obs).detach_batch_dim().0;
        let next_values = actor_critic.state_value(batch.next_obs).detach_batch_dim().0;
        let deltas = batch.reward.detach_batch_dim().0 + gamma * next_values * (1f32 - batch.terminated.clone().detach_batch_dim().0.float()) - values.clone();
        let terminated = batch.terminated;
        let truncated = batch.truncated;
        
        let mut adv = Tensor::zeros([n], &device).detach();
        adv = adv.slice_assign(n - n_envs..n, deltas.clone().slice(n - n_envs..n));
        for i in (0..n - n_envs).step_by(n_envs).rev() {
            adv = adv.clone().slice_assign(i..i + n_envs, deltas.clone().slice(i..i + n_envs) + gamma * lambda * adv.clone().slice(i + n_envs..i + 2 * n_envs) * (1f32 - truncated.clone().slice(0,i..i + 1).detach_batch_dim().0.float()) * (1f32 - terminated.clone().slice(0, i..i + 1).detach_batch_dim().0.float()));
        }
        let returns = adv.clone() + values;
        (adv.detach().attach_batch_dim(bs.clone()), returns.detach().attach_batch_dim(bs))
    }
}
