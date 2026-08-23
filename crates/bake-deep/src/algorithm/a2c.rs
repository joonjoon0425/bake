use burn::{Tensor, optim::{GradientsParams, ModuleOptimizer}};

use crate::{algorithm::dqn::ValueLoss, approximator::ActorCritic, config::ActorCriticConfig, distribution::Distribution, types::Batch, utils::gae};

pub struct A2C {
    pub gamma: f32,
    pub lambda: f32, // for GAE calculation
    pub value_loss: ValueLoss,
}

pub struct A2CLoss {
    pub actor_loss: Tensor<1>,
    pub critic_loss: Tensor<1>,
    pub entropy: Tensor<1>,
}

impl A2C {
    pub fn new(gamma: f32, lambda: f32, value_loss: ValueLoss) -> Self { Self { gamma, lambda, value_loss } }

    pub fn loss<Ac: ActorCritic>(config: &A2C, actor_critic: &Ac, batch: Batch<Ac::Obs, <Ac::Dist as Distribution>::Action, Ac::Constraint>) -> A2CLoss {
        let (dist, values) = actor_critic.forward(batch.obss, batch.constraints);
        let next_values = actor_critic.critic(batch.next_obss);
        let (adv, ret) = gae(batch.rewards, values.clone(), next_values, batch.terminated, batch.truncated, config.gamma, config.lambda);
        // 1. advantage
        let gae = (adv.clone() - adv.clone().mean()) / (adv.clone().var(0) + 1e-9).sqrt();
        // 2. policy surrogate
        let log_prob = dist.log_probs(batch.actions);
        let actor_loss = -(log_prob * gae).mean();
        // 3. entropy
        let entropy = dist.entropy().mean();
        // 4. value loss
        let critic_loss = config.value_loss.forward(values, ret.detach());

        A2CLoss { actor_loss, critic_loss, entropy }
    }

    pub fn update_separated<Ac: ActorCritic>(mut actor_critic: Ac, loss: A2CLoss, c_e: f32, lr_a: f64, opt_a: &mut ModuleOptimizer, lr_c: f64, opt_c: &mut ModuleOptimizer) -> Ac {
        let actor_loss = loss.actor_loss - loss.entropy * c_e;
        let grads = actor_loss.backward();
        let grads = GradientsParams::from_grads(grads, &actor_critic);
        actor_critic = opt_a.step(lr_a, actor_critic, grads);

        let grads = loss.critic_loss.backward();
        let grads = GradientsParams::from_grads(grads, &actor_critic);
        actor_critic = opt_c.step(lr_c, actor_critic, grads);

        actor_critic
    }
}