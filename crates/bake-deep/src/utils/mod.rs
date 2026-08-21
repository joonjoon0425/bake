use burn::{Tensor, tensor::TensorData};

use crate::types::Batchable;

/// A GAE compute function
pub fn gae(
    rewards: Tensor<1>,
    values: Tensor<1>,
    next_values: Tensor<1>,
    terminated: Tensor<1>,
    truncated: Tensor<1>,
    gamma: f32,
    lambda: f32,
) -> (Tensor<1>, Tensor<1>) {
    let n = rewards.batch_size();
    let device = rewards.device();
    let deltas = rewards + gamma * next_values * (1f32 - terminated.clone()) - values.clone();
    let deltas: Vec<f32> = deltas.into_data().into_vec().unwrap();
    let terminated: Vec<f32> = terminated.into_data().into_vec().unwrap();
    let truncated: Vec<f32> = truncated.into_data().into_vec().unwrap();

    let mut gae = vec![0f32; n];
    gae[n - 1] = deltas[n - 1];
    for i in (0..n-1).rev() {
        gae[i] = deltas[i] + gamma * lambda * gae[i + 1] * (1f32 - truncated[i]) * (1f32 - terminated[i]);
    }
    let gae = Tensor::from_data(TensorData::new(gae, [n]), &device);
    let returns = gae.clone() + values;
    (gae, returns)
}