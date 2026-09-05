//! An Advantage Actor-Critic algorithm implementation
use std::collections::HashMap;

use burn::{Tensor, optim::{GradientsParams, ModuleOptimizer}};

use crate::{algorithm::{advantage_enum::Advantage, loss_enum::Loss}, contract::ActorCritic, data::Batch, distribution::{Distribution, PossibleConstraint}, logger::ToLog};

/// state for A2C
#[derive(Debug, Clone)]
pub struct A2C {
    /// discount rate
    pub gamma: f32,
    /// for advantage calculation
    pub advantage: Advantage,
    /// loss function for critic
    pub loss_fn: Loss,
}

/// loss struct for A2C
#[derive(Debug, Clone)]
pub struct A2CLoss {
    /// loss of policy (actor)
    pub actor_loss: Tensor<1>,
    /// loss of value (critic)
    pub critic_loss: Tensor<1>,
    /// entropy
    pub entropy: Tensor<1>,
}

impl A2C {
    /// compute the loss of A2C
    pub fn loss<Ac: ActorCritic>(state: &A2C, actor_critic: &Ac, batch: Batch<Ac::Obs, <Ac::Dist as Distribution>::Sample, impl PossibleConstraint<Ac::Dist>>) -> A2CLoss {
        let (dist, values) = actor_critic.forward(batch.obss.clone(), batch.constraints.clone());
        let (adv, ret) = state.advantage.advantage(actor_critic, batch.clone(), state.gamma);
        // 1. advantage
        let adv = (adv.clone() - adv.clone().mean()) / (adv.clone().var(0) + 1e-9).sqrt();
        // 2. policy surrogate
        let log_prob = dist.log_probs(batch.actions);
        let actor_loss = -(log_prob * adv).mean();
        // 3. entropy
        let entropy = dist.entropy().mean();
        // 4. value loss
        let critic_loss = state.loss_fn.forward(values, ret);

        A2CLoss { actor_loss, critic_loss, entropy }
    }

    /// update the network.
    /// # Warning
    /// This update is for encoder-separated actor-critics
    pub fn update_separated<Ac: ActorCritic>(actor_critic: Ac, loss: A2CLoss, c_e: f32, lr_a: f64, opt_a: &mut ModuleOptimizer, lr_c: f64, opt_c: &mut ModuleOptimizer) -> Ac {
        assert!(actor_critic.encoder_type() == crate::contract::actor_critic::EncoderType::Separated, "The update_separated cannot be called with encoder-sharing actor critic");
        let actor_loss = loss.actor_loss - loss.entropy * c_e;
        let grads = actor_loss.backward();
        let grads = GradientsParams::from_grads(grads, &actor_critic);
        let actor_critic = opt_a.step(lr_a, actor_critic, grads);

        let grads = loss.critic_loss.backward();
        let grads = GradientsParams::from_grads(grads, &actor_critic);

        opt_c.step(lr_c, actor_critic, grads)
    }

    /// update the network.
    /// # Warning
    /// This update is for encoder-shared actor-critics
    pub fn update_shared<Ac: ActorCritic>(actor_critic: Ac, loss: A2CLoss, c_e: f32, c_c: f32, lr: f64, opt: &mut ModuleOptimizer) -> Ac {
        assert!(actor_critic.encoder_type() == crate::contract::actor_critic::EncoderType::Shared, "The update_shared cannot be called with encoder-separated actor critic");
        let loss = loss.actor_loss - loss.entropy * c_e + loss.critic_loss * c_c;
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &actor_critic);
        
        opt.step(lr, actor_critic, grads)
    }

    /// gives the name of recordable logs. use it to register at the logger
    pub fn log_names() -> Vec<&'static str> {
        vec![
            "actor_loss",
            "critic_loss",
            "entropy"
        ]
    }
}

impl ToLog for A2CLoss {
    fn to_log(&self) -> HashMap<&'static str, f32> {
        let mut record = HashMap::new();
        record.insert("actor_loss", self.actor_loss.clone().into_scalar());
        record.insert("critic_loss", self.critic_loss.clone().into_scalar());
        record.insert("entropy", self.entropy.clone().into_scalar());
        record
    }
}
