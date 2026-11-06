//! Advantage estimator
use burn::prelude::*;
use crate::deep::contract::ActorCritic;
use crate::deep::distribution::{Distribution, PossibleConstraint};
use crate::core::data::{Batch, BatchOps, Transition};

/// advantage estimation
#[derive(Debug, Clone)]
pub enum AdvantageEstimator {
    /// 1-step TD residual
    Td0,
    /// Monte Carlo residual
    Td1,
    /// Generalized Advantge Estimation
    Gae {
        /// controls bias-variance tradeoff
        /// 0 -> low variance, high bias
        /// 1 -> high variance, low bias
        lambda: f32,
    }
}

impl AdvantageEstimator {
    /// compute the advantage
    /// # Returns
    /// - (advantage, return) with shape [T * N]
    pub fn advantage<Ac: ActorCritic>(
        &self,
        ac: &Ac,
        t: Batch<Transition<crate::core::Burn, Ac::Obs, <Ac::Dist as Distribution>::Sample, impl PossibleConstraint<Ac::Dist>, Tensor<1>>>,
        gamma: f32
    ) -> (Tensor<1>, Tensor<1>) {
        match self {
            AdvantageEstimator::Td0 => Self::td0(ac, t, gamma),
            AdvantageEstimator::Td1 => Self::td1(ac, t, gamma),
            AdvantageEstimator::Gae { lambda } => Self::gae(ac, t, gamma, *lambda)
        }
    }

    /// compute the normalized advantage
    /// # Returns
    /// - (normalized advantage, return) with shape [T * N]
    pub fn normalized_advantage<Ac: ActorCritic>(
        &self,
        ac: &Ac,
        t: Batch<Transition<crate::core::Burn, Ac::Obs, <Ac::Dist as Distribution>::Sample, impl PossibleConstraint<Ac::Dist>, Tensor<1>>>,
        gamma: f32
    ) -> (Tensor<1>, Tensor<1>) {
        let (adv, ret) = self.advantage(ac, t, gamma);
        let adv = (adv.clone() - adv.clone().mean()) / (adv.var(0).sqrt() + 1e-9);
        (adv, ret)
    }

    fn td0<Ac: ActorCritic>(
        ac: &Ac,
        t: Batch<Transition<crate::core::Burn, Ac::Obs, <Ac::Dist as Distribution>::Sample, impl PossibleConstraint<Ac::Dist>, Tensor<1>>>,
        gamma: f32
    ) -> (Tensor<1>, Tensor<1>) {
        let t = t.destruct().0;
        let values = ac.state_value(t.obs);
        let next_values = ac.state_value(t.next_obs);
        let adv = t.reward + gamma * next_values * (1f32 - t.terminated.float()) - values.clone();

        let ret = adv.clone() + values;
        (adv.detach(), ret.detach())
    }

    fn td1<Ac: ActorCritic>(
        ac: &Ac,
        t: Batch<Transition<crate::core::Burn, Ac::Obs, <Ac::Dist as Distribution>::Sample, impl PossibleConstraint<Ac::Dist>, Tensor<1>>>,
        gamma: f32
    ) -> (Tensor<1>, Tensor<1>) {
        Self::gae(ac, t, gamma, 1.)
    }

    fn gae<Ac: ActorCritic>(
        ac: &Ac,
        t: Batch<Transition<crate::core::Burn, Ac::Obs, <Ac::Dist as Distribution>::Sample, impl PossibleConstraint<Ac::Dist>, Tensor<1>>>,
        gamma: f32, lambda: f32
    ) -> (Tensor<1>, Tensor<1>) {
        let device = t.device();
        let (t, bs) = t.destruct();
        let values = ac.state_value(t.obs).detach();
        let next_values = ac.state_value(t.next_obs).detach();

        let deltas =
            (t.reward + gamma * next_values * (1f32 - t.terminated.clone().float()) - values.clone())
            .unbind_batch(&bs, 0);
        let conts =
            ((1f32 - t.terminated.float()) * (1f32 - t.truncated.float()))
            .unbind_batch(&bs, 0);
        let mut advs = Vec::with_capacity(deltas.len());
        let mut next = Tensor::zeros([bs[1]], &device);
        
        for (delta, cont) in deltas.into_iter().zip(conts.into_iter()).rev() {
            next = delta + gamma * lambda * cont * next;
            advs.push(next.clone());
        }
        advs.reverse();

        let adv = BatchOps::cat_batch(&vec![vec![1, bs[1]]; bs[0]], 0, advs);
        let ret = adv.clone() + values;
        (adv.detach(), ret.detach())
    }
}