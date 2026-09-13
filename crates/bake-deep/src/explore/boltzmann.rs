//! boltzmann exploration
//! 
use burn::{prelude::*, tensor::activation::softmax};

use crate::{constraint::discrete_constraint::DiscreteConstraint, contract::DiscreteQFunction, explore::Exploration};

/// Boltzmann (softmax) policy implementation
pub struct Boltzmann {
    temp: f32,
}

impl Boltzmann {
    /// create a new `Boltzmann` policy. The higher the temperature, more uniform-like the policy.
    pub fn new(temp: f32) -> Self {
        if temp < 0.0 { panic!("the temp must be bigger than 0.0. Current temp is {}", temp) } 
        Self {
            temp,
        }
    }

    /// get current temperature of Boltzmann policy
    pub fn temp(&self) -> f32 { self.temp }

    /// get the mutable reference of temperature of Boltzmann policy
    pub fn temp_mut(&mut self) -> &mut f32 { &mut self.temp }
}

impl Exploration for Boltzmann {
    fn sample<Q: DiscreteQFunction>(&mut self, qfunc: &Q, obs: Q::Obs, constraint: impl DiscreteConstraint) -> Tensor<1, Int> {
        let probs = softmax(qfunc.forward(obs, constraint) / self.temp, 1);
        let actions = probs.categorical(1).squeeze_dim(1);
        actions
    }
}

#[cfg(test)]
mod tests {
    use crate::{constraint::discrete_constraint::DiscreteMask, explore::{Boltzmann, Exploration}, net::basic::MlpDiscreteQNet, wrapper::DiscreteQNetWrapper};
    use burn::{nn::activation::ActivationConfig::Relu, prelude::*};
    
    #[test]
    #[should_panic]
    fn wrong_temp() {
        Boltzmann::new(-1.0);
    }

    #[test]
    fn mask_test() {
        let device = Device::default();
        device.seed(12);
        let c = DiscreteMask(Tensor::from_bool([[false, false, true, false], [false, false, false, true], [true, false, false, false]], &device));
        let mut e = Boltzmann::new(1.0);
        let qnet = DiscreteQNetWrapper::new(MlpDiscreteQNet::new(&[1, 1, 4], Relu, &device));
        let obs = Tensor::from_floats([[11.], [-1.], [2.]], &device);
        let action = e.sample(&qnet, obs, c);
        assert!(action.equal(Tensor::from_ints([2, 3, 0], &device)).all().into_scalar::<bool>());
    }
}