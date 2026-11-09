//! Advantage Actor-Critic
use burn::prelude::*;
use crate::core::data::{Shaped, BatchOps, BatchedTransition};
use crate::deep::distribution::{Distribution, PossibleConstraint};
use crate::deep::algorithms::advantage_estimator::AdvantageEstimator;
use crate::deep::loss::{ActorCriticLoss, Loss, LossFn};
use crate::deep::contract::ActorCritic;

/// state struct for Advantage Actor Critic algorithm
#[derive(Debug, Clone)]
pub struct A2CState {
    /// discount rate
    pub gamma: f32,
    /// entropy coefficient
    pub c_e: f32,
    /// critic coefficient
    /// # Warning
    /// - The critic coefficient will not be used when the actor and critic does not share the encoder
    pub c_c: f32,
    /// for advantage calculation
    pub advantage: AdvantageEstimator,
    /// loss function for critic
    pub loss_fn: LossFn,
}

/// loss struct for A2C
#[derive(Debug, Clone)]
pub struct A2CLoss {
    /// loss
    pub loss: Tensor<1>,
    /// policy loss with entropy
    pub policy_loss_with_entropy: Tensor<1>,
    /// policy loss
    pub policy_loss: Tensor<1>,
    /// value loss
    pub value_loss: Tensor<1>,
    /// entropy
    pub entropy: Tensor<1>,
}

impl Loss for A2CLoss {
    fn loss(&self) -> Tensor<1> {
        self.loss.clone()
    }
}

impl ActorCriticLoss for A2CLoss {
    fn policy_loss_with_entropy(&self) -> Tensor<1> {
        self.policy_loss_with_entropy.clone()
    }

    fn value_loss(&self) -> Tensor<1> {
        self.value_loss.clone()
    }
}

/// compute the loss of A2C
/// # Warning
/// - Here, the given ActorCritic is moved to autodiff device
/// - The ActorCritic will be move to inner device when `update` is called
pub fn a2c_loss<Ac: ActorCritic>(
    state: &A2CState,
    ac: Ac,
    batch: Shaped<BatchedTransition<Ac::Obs, impl PossibleConstraint<Ac::Dist>, <Ac::Dist as Distribution>::Sample, Tensor<1>>>
) -> (Ac, A2CLoss) {
    let ac = ac.train();
    let batch = batch.autodiff();
    let (dist, value) = ac.dist_and_state_value(batch.obs.clone(), batch.constraint.clone());
    // 1. compute advantage and get policy loss
    let log_prob = dist.log_probs(batch.action.clone());

    let (adv, ret) = state.advantage.normalized_advantage(&ac, batch, state.gamma);
    
    let policy_loss = -(log_prob * adv).mean();
    // 2. compute entropy
    let entropy = dist.entropy().mean();
    // 3. compute value loss
    let value_loss = state.loss_fn.forward(value, ret);
    // 4. compute loss
    let policy_loss_with_entropy = policy_loss.clone() - state.c_e * entropy.clone();
    let loss = policy_loss_with_entropy.clone() + state.c_c * value_loss.clone();

    (ac, A2CLoss { loss, policy_loss_with_entropy, policy_loss, value_loss, entropy })
}