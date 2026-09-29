//! A REINFORCE algorithm implementation
//! 

use bake_common::logger::ToLog;
use burn::prelude::*;

use crate::{
    contract::basic::Policy, data::{Batch, Batchable}, distribution::{Distribution, PossibleConstraint}, loss::traits::TotalLoss
};

/// A state of REINFORCE algorithm
#[derive(Debug, Clone)]
pub struct Reinforce {
    /// discount rate
    pub gamma: f32,
    /// entropy coeffiecient
    pub c_e: f32,
    /// baseline for computing the advantage
    pub baseline: Baseline,
}

impl Reinforce {
    /// compute the loss of REINFORCE algorithm
    /// # Warning
    /// - Here, the given Policy is moved to autodiff device
    /// - The Policy will be move to inner device when `update` is called
    pub fn loss<P: Policy>(state: &Reinforce, policy: P, rollout: Batch<P::Obs, <P::Dist as Distribution>::Sample, impl PossibleConstraint<P::Dist>>) -> (P, ReinforceLoss) {
        let policy = policy.train();
        let rollout = rollout.into_autodiff();

        let len = rollout.batch_size().unwrap();
        let device = rollout.device();
        let dist = policy.dist(rollout.obss, rollout.constraints);
        let mut returns = Tensor::zeros([len], &device);
        returns.assign_inplace(rollout.rewards.clone().slice(len - 1..len), len - 1);
        for i in (0..(len - 1)).rev() {
            let r = rollout.rewards.clone().slice(i..i + 1) + state.gamma * returns.clone().slice(i + 1..i + 2);
            returns.assign_inplace(r, i);
        }

        let returns = state.baseline.advantage(returns).detach();
        let log_probs = dist.log_probs(rollout.actions);
        let entropy = dist.entropy().mean();
        let policy_loss = -(returns * log_probs).mean();

        let total_loss = policy_loss.clone() - state.c_e * entropy.clone();
        (policy, ReinforceLoss { total_loss, policy_loss, entropy })
    }

    /// gives the name of recordable logs. use it to register at the logger
    pub fn log_names() -> Vec<&'static str> {
        vec![
            "loss",
            "policy_loss",
            "entropy"
        ]
    }
}

/// loss struct for REINFORCE algorithm
#[derive(Debug, Clone)]
pub struct ReinforceLoss {
    /// total loss (policy loss with entropy)
    pub total_loss: Tensor<1>,
    /// the policy loss of REINFORCE algorithm
    pub policy_loss: Tensor<1>,
    /// the entropy of policy
    pub entropy: Tensor<1>,
}

impl TotalLoss for ReinforceLoss {
    fn total_loss(&self) -> Tensor<1> {
        self.total_loss.clone()
    }
}

impl ToLog for ReinforceLoss {
    fn to_log(&self) -> std::collections::HashMap<&'static str, f32> {
        let mut record = std::collections::HashMap::new();
        record.insert("loss", self.total_loss.clone().into_scalar());
        record.insert("policy_loss", self.policy_loss.clone().into_scalar());
        record.insert("entropy", self.entropy.clone().into_scalar());

        record
    }
}

/// baseline enum for REINFORCE algorithm
#[derive(Debug, Config)]
pub enum Baseline {
    /// No baseline is applied
    None,
    /// Normalize the return
    Normalized,
    /// Set the mean of returns into zero
    Mean,
}

impl Baseline {
    /// calculate the return with baseline applied
    pub fn advantage(&self, returns: Tensor<1>) -> Tensor<1> {
        match self {
            Baseline::None => returns,
            Baseline::Normalized => {
                (returns.clone() - returns.clone().mean()) / (returns.var(0).sqrt() + 1e-8)
            }
            Baseline::Mean => {
                returns.clone() - returns.clone().mean()
            }
        }
    }
}