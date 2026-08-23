use bake_macros::Batchable;
use burn::{Tensor, tensor::Int};

use crate::algorithm::dqn::ValueLoss;
use crate::{distribution::Distribution, approximator::ActorCritic, types::{Batch, Batchable}};


pub struct Ppo {
    pub gamma: f32,
    pub lambda: f32,
    pub eps: f32,
    pub value_loss: ValueLoss,
}

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
    pub fn new(gamma: f32, lambda: f32, eps: f32, value_loss: ValueLoss) -> Self { Self { gamma, lambda, eps, value_loss } }

    pub fn loss<Ac: ActorCritic>(config: &Ppo, actor_critic: &Ac, minibatch: Batch<Ac::Obs, <Ac::Dist as Distribution>::Action, Ac::Constraint, PpoExtra>) -> PpoLoss {
        let (dist, values) = actor_critic.forward(minibatch.obss, minibatch.constraints);
        let log_ratio = dist.log_probs(minibatch.actions) - minibatch.extras.old_log_probs;
        let ratio = log_ratio.clone().exp();
        let stacked = Tensor::stack::<2>(vec![ratio.clone() * minibatch.extras.gae.clone(), ratio.clone().clamp(1f32 - config.eps, 1f32 + config.eps) * minibatch.extras.gae ], 1);
        let actor_loss = -stacked.min_dim(1).mean();
        let critic_loss = config.value_loss.forward(values.clone(), minibatch.extras.ret.detach());
        let entropy = dist.entropy().mean();

        let approx_kl = ((log_ratio.clone().exp() - 1f32) - log_ratio.clone()).mean();
        let clip_ratio = (log_ratio.exp() - 1f32).abs().greater_elem(config.eps).float().mean();

        PpoLoss { actor_loss, critic_loss, entropy, approx_kl, clip_ratio }
    }
}

/// logging struct for PPO
#[derive(Debug, Clone, Default)]
pub struct PPOLog {
    /// entropy
    pub entropy: Option<Tensor<1>>,
    /// approximate KL divergence
    pub approx_kl: Option<Tensor<1>>,
    /// clipped ratio
    pub clip_ratio: Option<Tensor<1>>,
}
