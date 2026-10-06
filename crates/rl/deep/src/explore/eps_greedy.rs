//! epsilon-greedy policy
//!
use bake_rl_core::deep::constraint::discrete_constraint::DiscreteConstraint;
use burn::{Tensor, tensor::{Distribution, Int}};
use rand::{RngExt, SeedableRng, rngs::SmallRng};

use crate::{contract::basic::DiscreteActionValue, explore::Exploration};

/// An epsilon-greedy policy implementation
pub struct EpsGreedy {
    eps: f32,
    rng: SmallRng,
}

impl EpsGreedy {
    /// Create a new `EpsGreedy` policy. The epsilon is the probability of choosing action randomly
    pub fn new(seed: u64, eps: f32) -> Self {
        if eps < 0.0 || eps > 1.0 { panic!("the eps must be between 0.0 and 1.0. Current eps is {}", eps) }
        Self {
            eps,
            rng: SmallRng::seed_from_u64(seed)
        }
    }

    /// get the value of epsilon
    pub fn eps(&self) -> f32 { self.eps }

    /// get the mutable reference of epsilon
    pub fn eps_mut(&mut self) -> &mut f32 { &mut self.eps }
}

impl Exploration for EpsGreedy {
    /// sample an action from given Q values.
    fn sample<Q: DiscreteActionValue>(&mut self, qfunc: &Q, obs: Q::Obs, constraint: impl DiscreteConstraint) -> Tensor<1, Int> {
        let qvalues = qfunc.action_values(obs, constraint.clone());
        if self.rng.random_range(0.0..1.0) < self.eps {
            let random = Tensor::random_like(&qvalues, Distribution::Default);
            constraint.apply(random, -1f32).argmax(1).squeeze_dim(1)
        } else {
            qvalues.argmax(1).squeeze_dim(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use bake_rl_core::deep::constraint::discrete_constraint::DiscreteMask;
use burn::{nn::activation::ActivationConfig::Relu, prelude::*};
    use crate::{explore::{EpsGreedy, Exploration}, net::basic::MlpDiscreteQNet};

    #[test]
    #[should_panic]
    fn wrong_eps_over_one() {
        EpsGreedy::new(0, 2.0);
    }

    #[test]
    #[should_panic]
    fn wrong_eps_below_zero() {
        EpsGreedy::new(0, -2.0);
    }

    #[test]
    fn mask_test() {
        let device = Device::default();
        device.seed(12);
        let c = DiscreteMask(Tensor::from_bool([[false, false, true, false], [false, false, false, true], [true, false, false, false]], &device));
        let mut e = EpsGreedy::new(1, 1.0);
        let qnet = MlpDiscreteQNet::new(&[1, 1, 4], Relu, &device);
        let obs = Tensor::from_floats([[11.], [-1.], [2.]], &device);
        let action = e.sample(&qnet, obs, c);
        assert!(action.equal(Tensor::from_ints([2, 3, 0], &device)).all().into_scalar::<bool>());
    }
}