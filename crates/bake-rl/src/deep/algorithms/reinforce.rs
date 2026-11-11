//! Reinforce (Vanilla Policy Grandient)
use burn::prelude::*;
use crate::core::data::{Shaped, BatchOps, BatchedTransition};
use crate::deep::distribution::{Distribution, PossibleConstraint};
use crate::deep::loss::Loss;
use crate::deep::contract::Policy;

/// state struct for Reinforce algorithm
#[derive(Debug, Clone)]
pub struct REINFORCEState {
    /// discount rate
    pub gamma: f32,
    /// entropy coefficient
    pub c_e: f32,
    /// baseline
    pub baseline: REINFORCEBaseline,
}

/// loss struct for Reinforce
#[derive(Debug, Clone)]
pub struct REINFORCELoss {
    /// (loss) policy loss with entropy
    pub loss: Tensor<1>,
    /// policy loss (no autodiff)
    pub policy_loss: Tensor<1>,
    /// entropy (no autodiff)
    pub entropy: Tensor<1>,
}

impl Loss for REINFORCELoss {
    fn loss(&self) -> Tensor<1> {
        self.loss.clone()
    }
}

/// baseline enumeration for Reinforce
#[derive(Debug, Clone, Copy)]
pub enum REINFORCEBaseline {
    /// No baseline is applied
    None,
    /// leave-one-out baseline (means computed each time step)
    LeaveOneOut,
}

impl REINFORCEBaseline {
    /// calculate the return with baseline applied, with masks
    /// - expects `return` to have shape [T, N]
    pub fn baseline(&self, returns: Tensor<2>, valid: Tensor<2, Bool>) -> Tensor<2> {
        match self {
            Self::None => returns,
            Self::LeaveOneOut => {
                assert!(returns.dims()[1] != 1, "cannot use `REINFORCEBaseline::LeaveOneOut` with single environment");
                let sum_along_env = returns.clone().mask_fill(valid.clone().bool_not(), 0).sum_dim(1);
                let k = valid.clone().float().sum_dim(1);
                let k_minus_1 = k - valid.clone().float();
                let leave_one_out = (sum_along_env - returns.clone()) / k_minus_1.clone();
                let leave_one_out = leave_one_out.mask_fill(k_minus_1.equal_scalar(0), 0);
                (returns - leave_one_out).mask_fill(valid.bool_not(), 0)
            }
        }
    }
}


/// compute the loss of Reinforce with reward-to-go
/// # Warning
/// - The algorithms regards truncation as termination
/// - The algorithm requires all trajectories to start at time step 0 and has at least one complete trajectory in `batch`
///     - The algorithm uses only the first episode for loss computation
///     - Using all trajectories makes bias toward short trajectories, since the stopping time itself depends on trajectories
///     - (personal message: I'd study stochastic process and markov decision processes and some mathematically rigurous theories later...)
/// - Here, the given `Policy` is moved to autodiff device
/// - Then Policy will be moved to inner device when `update` is called
pub fn reinforce_loss<P: Policy>(
    state: &REINFORCEState,
    policy: P,
    batch: Shaped<BatchedTransition<P::Obs, impl PossibleConstraint<P::Dist>, <P::Dist as Distribution>::Sample, Tensor<1>>>
) -> (P, REINFORCELoss) {
    let device = batch.device();
    let policy = policy.train();
    let (batch, bs) = batch.autodiff().destruct();
    let dist = policy.dist(batch.obs, batch.constraint);

    // make a mask
    // `done` for first time
    // reshape into [T, N] first
    let ter = batch.terminated.reshape([bs[0], bs[1]]);
    let tru = batch.truncated.reshape([bs[0], bs[1]]);
    let done = ter.clone().bool_or(tru).int();
    // 1 for first eposides and 0 for later trajectories
    let valid = (done.clone().cumsum(0) - done).equal_elem(0);

    let rewards = 
        batch.reward.unbind_batch(&bs, 0);
    let valids = 
        valid.clone().flatten::<1>(0, -1).unbind_batch(&bs, 0);

    let mut returns = Vec::with_capacity(bs[0]);
    let mut next = Tensor::zeros([bs[1]], &device);

    for (reward, valid) in rewards.into_iter().zip(valids.into_iter()).rev() {
        next = (reward + state.gamma * next) * valid.float();
        returns.push(next.clone());
    }
    returns.reverse();

    let returns = Tensor::stack::<2>(returns, 0);
    // apply baseline: R(\tau) - baseline
    let returns = state.baseline.baseline(returns, valid.clone());
    // gamma^t
    let discounts = Tensor::<1>::from_floats([state.gamma], &device).powf(Tensor::arange(0..bs[0] as i64, &device).float());
    let log_probs = dist.log_probs(batch.action);
    // entropy (sum divided by n_envs, since the poliy is computed in that way. ppo and a2c just calls `mean`)
    let entropy = (dist.entropy() * valid.flatten(0, -1).float()).sum() / bs[1] as i64;
    let policy_loss = -((returns * discounts.unsqueeze_dim(1)).flatten(0, -1) * log_probs).sum() / bs[1] as i64;

    let loss = policy_loss.clone() - state.c_e * entropy.clone();
    (policy, REINFORCELoss { loss, policy_loss: policy_loss.inner(), entropy: entropy.inner() })
}