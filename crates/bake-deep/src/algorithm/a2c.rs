use std::collections::HashMap;

use burn::{Tensor, config::Config, optim::{GradientsParams, ModuleOptimizer}};

use crate::{algorithm::dqn::ValueLoss, approximator::ActorCritic, distribution::Distribution, types::{Batch, Recordable}, utils::gae};

#[derive(Debug, Config)]
pub struct A2C {
    pub gamma: f32,
    pub lambda: f32, // for GAE calculation
    pub value_loss: ValueLoss,
}

#[derive(Debug, Config)]
pub struct A2CLoss {
    pub actor_loss: Tensor<1>,
    pub critic_loss: Tensor<1>,
    pub entropy: Tensor<1>,
}

impl A2C {
    pub fn loss<Ac: ActorCritic>(config: &A2C, actor_critic: &Ac, batch: Batch<Ac::Obs, <Ac::Dist as Distribution>::Action, Ac::Constraint>) -> A2CLoss {
        let (dist, values) = actor_critic.forward(batch.obss.clone(), batch.constraints.clone());
        let (adv, ret) = gae(actor_critic, batch.clone(), config.gamma, config.lambda);
        // 1. advantage
        let gae = (adv.clone() - adv.clone().mean()) / (adv.clone().var(0) + 1e-9).sqrt();
        // 2. policy surrogate
        let log_prob = dist.log_probs(batch.actions);
        let actor_loss = -(log_prob * gae).mean();
        // 3. entropy
        let entropy = dist.entropy().mean();
        // 4. value loss
        let critic_loss = config.value_loss.forward(values, ret);

        A2CLoss { actor_loss, critic_loss, entropy }
    }

    pub fn update_separated<Ac: ActorCritic>(actor_critic: Ac, loss: A2CLoss, c_e: f32, lr_a: f64, opt_a: &mut ModuleOptimizer, lr_c: f64, opt_c: &mut ModuleOptimizer) -> Ac {
        assert!(!actor_critic.shares_encoder(), "The update_separated cannot be called with encoder-sharing actor critic");
        let actor_loss = loss.actor_loss - loss.entropy * c_e;
        let grads = actor_loss.backward();
        let grads = GradientsParams::from_grads(grads, &actor_critic);
        let actor_critic = opt_a.step(lr_a, actor_critic, grads);

        let grads = loss.critic_loss.backward();
        let grads = GradientsParams::from_grads(grads, &actor_critic);

        opt_c.step(lr_c, actor_critic, grads)
    }

    pub fn update_shared<Ac: ActorCritic>(actor_critic: Ac, loss: A2CLoss, c_e: f32, c_c: f32, lr: f64, opt: &mut ModuleOptimizer) -> Ac {
        assert!(actor_critic.shares_encoder(), "The update_shared cannot be called with encoder-separated actor critic");
        let loss = loss.actor_loss - loss.entropy * c_e + loss.critic_loss * c_c;
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &actor_critic);
        
        opt.step(lr, actor_critic, grads)
    }
}

impl Recordable for A2CLoss {
    fn to_record(&self) -> HashMap<&'static str, Tensor<1>> {
        let mut record = HashMap::new();
        record.insert("actor_loss", self.actor_loss.clone().detach());
        record.insert("critic_loss", self.critic_loss.clone().detach());
        record.insert("entropy", self.entropy.clone().detach());
        record
    }
}