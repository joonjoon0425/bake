use burn::{Tensor, tensor::TensorData};

use crate::{approximator::ActorCritic, distribution::Distribution, types::{Batch, Batchable}};

/// A GAE compute function
pub fn gae<Ac: ActorCritic, Extra: Batchable>(
    actor_critic: &Ac,
    batch: Batch<Ac::Obs, <Ac::Dist as Distribution>::Action, Ac::Constraint, Extra>,
    gamma: f32,
    lambda: f32,
) -> (Tensor<1>, Tensor<1>) {
    let n = batch.batch_size();
    let device = batch.rewards.device();
    let values = actor_critic.critic(batch.obss);
    let next_values = actor_critic.critic(batch.next_obss);
    let deltas = batch.rewards + gamma * next_values * (1f32 - batch.terminated.clone()) - values.clone();
    let deltas: Vec<f32> = deltas.into_data().into_vec().unwrap();
    let terminated: Vec<f32> = batch.terminated.into_data().into_vec().unwrap();
    let truncated: Vec<f32> = batch.truncated.into_data().into_vec().unwrap();

    let mut adv = vec![0f32; n];
    adv[n - 1] = deltas[n - 1];
    for i in (0..n-1).rev() {
        adv[i] = deltas[i] + gamma * lambda * adv[i + 1] * (1f32 - truncated[i]) * (1f32 - terminated[i]);
    }
    let adv = Tensor::from_data(TensorData::new(adv, [n]), &device);
    let returns = adv.clone() + values;
    (adv.detach(), returns.detach())
}