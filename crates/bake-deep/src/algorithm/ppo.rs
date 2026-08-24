use std::collections::HashMap;

use bake_macros::Batchable;
use burn::config::Config;
use burn::optim::{GradientsParams, ModuleOptimizer};
use burn::{Tensor, tensor::Int};

use crate::algorithm::dqn::ValueLoss;
use crate::types::Recordable;
use crate::{distribution::Distribution, approximator::ActorCritic, types::{Batch, Batchable}};

#[derive(Debug, Config)]
pub struct Ppo {
    pub gamma: f32,
    pub lambda: f32,
    pub eps: f32,
    pub value_loss: ValueLoss,
}

#[derive(Debug, Clone)]
pub struct PpoLoss {
    pub actor_loss: Tensor<1>,
    pub critic_loss: Tensor<1>,
    pub entropy: Tensor<1>,

    pub approx_kl: Tensor<1>,
    pub clip_ratio: Tensor<1>,
}

#[derive(Batchable, Debug, Clone)]
pub struct PpoExtra {
    pub gae: Tensor<1>,
    pub ret: Tensor<1>,
    pub old_log_probs: Tensor<1>,
}

impl Ppo {
    pub fn loss<Ac: ActorCritic>(config: &Ppo, actor_critic: &Ac, minibatch: Batch<Ac::Obs, <Ac::Dist as Distribution>::Action, Ac::Constraint, PpoExtra>) -> PpoLoss {
        let (dist, values) = actor_critic.forward(minibatch.obss, minibatch.constraints);
        let log_ratio = dist.log_probs(minibatch.actions) - minibatch.extras.old_log_probs;
        let ratio = log_ratio.clone().exp();
        let stacked = Tensor::stack::<2>(vec![ratio.clone() * minibatch.extras.gae.clone(), ratio.clone().clamp(1f32 - config.eps, 1f32 + config.eps) * minibatch.extras.gae ], 1);
        let actor_loss = -stacked.min_dim(1).mean();
        let critic_loss = config.value_loss.forward(values.clone(), minibatch.extras.ret.detach());
        let entropy = dist.entropy().mean();

        let approx_kl = ((log_ratio.clone().exp() - 1f32) - log_ratio.clone()).mean().detach();
        let clip_ratio = (log_ratio.exp() - 1f32).abs().greater_elem(config.eps).float().mean().detach();

        PpoLoss { actor_loss, critic_loss, entropy, approx_kl, clip_ratio }
    }

    pub fn update_separated<Ac: ActorCritic>(actor_critic: Ac, loss: PpoLoss, c_e: f32, lr_a: f64, opt_a: &mut ModuleOptimizer, lr_c: f64, opt_c: &mut ModuleOptimizer) -> Ac {
        let actor_loss = loss.actor_loss - loss.entropy * c_e;
        let grads = actor_loss.backward();
        let grads = GradientsParams::from_grads(grads, &actor_critic);
        let actor_critic = opt_a.step(lr_a, actor_critic, grads);

        let grads = loss.critic_loss.backward();
        let grads = GradientsParams::from_grads(grads, &actor_critic);

        opt_c.step(lr_c, actor_critic, grads)
    }

    pub fn update_shared<Ac: ActorCritic>(actor_critic: Ac, loss: PpoLoss, c_e: f32, c_c: f32, lr: f64, opt: &mut ModuleOptimizer) -> Ac {
        let loss = loss.actor_loss - loss.entropy * c_e + loss.critic_loss * c_c;
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &actor_critic);
        
        opt.step(lr, actor_critic, grads)
    }
}

impl Recordable for PpoLoss {
    fn to_record(&self) -> HashMap<&'static str, Tensor<1>> {
        let mut record = HashMap::new();
        record.insert("actor_loss", self.actor_loss.clone().detach());
        record.insert("critic_loss", self.critic_loss.clone().detach());
        record.insert("entropy", self.entropy.clone().detach());
        record.insert("approx_kl", self.approx_kl.clone().detach());
        record.insert("clip_ratio", self.clip_ratio.clone().detach());
        record
    }
}