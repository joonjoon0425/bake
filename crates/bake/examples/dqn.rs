use bake_deep::{agent::DQNAgent, buffer::ReplayBuffer, env::{CartPole, Env}, policy::EpsGreedy, types::{DuelingQNetwork, MLPQNetwork, Transition}};
use burn::{Tensor, nn::Relu, optim::AdamConfig, tensor::{Device}};

pub fn main() {
    let device = Device::default().autodiff();
    device.seed(12);
    let mut env = CartPole::new(12, &device);
    let mut agent = DQNAgent::new(0.99,    
        MLPQNetwork::new(vec![4, 128, 2], burn::nn::activation::Activation::Relu(Relu), &device),
        1e-3,
            AdamConfig::new().init()
        );
    let mut policy = EpsGreedy::new(123, 1.0f32);
    let mut buffer = ReplayBuffer::new(12, 10000, device.clone());
    let mut count = 0;
    for episode in 0..=4000 {
        let (mut obs, mut mask) = env.reset();
        let mut steps = 0;
        
        let mut loss: Tensor<1> = Tensor::zeros([1], &device);
        let mut td_error: Tensor<1> = Tensor::zeros([1], &device);
        let mut q_mean: Tensor<1> = Tensor::zeros([1], &device);

        loop {
            let action = agent.action(&mut policy, obs.clone(), mask.clone());
            let ((next_obs, next_mask), reward, terminated, truncated) = env.step(action);

            buffer.push(Transition {
                obs,
                action,
                reward,
                next_obs: next_obs.clone(),
                terminated,
                truncated,
                mask,
                next_mask: next_mask.clone(),
                extra: (),
            });

            if let Some(batch) = buffer.sample(64) {
                (agent, q_mean, loss, td_error) = agent.update(batch);
            }

            if count % 1000 == 0 {
                agent.sync();
            }

            obs = next_obs;
            mask = next_mask;
            count += 1;
            steps += 1;
            if terminated || truncated { break; }
        }
        *policy.eps_mut() *= 0.99;
        if episode % 100 == 0 { println!("episode: {episode}, steps: {steps}, loss: {}, q_mean: {}, eps: {}", loss.into_scalar::<f32>(), q_mean.into_scalar::<f32>(), policy.eps()); }
    }
}