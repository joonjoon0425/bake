use bake_deep::{agent::{PPOAgent}, buffer::RolloutBuffer, config::ActorCriticConfig, distribution::Distribution, encoder::MLPEncoder, env::CartPole, head::{CategoricalHead, LinearVHead}, network::SequentialActorCriticNetwork, types::Tape};
use burn::{Tensor, nn::activation::Activation, optim::{RmsPropConfig}, tensor::Device};


pub fn main() {
    let seed: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(12);
    let device = Device::default().autodiff();
    device.seed(seed);
    println!("# optimizer=rmsprop lr_p=1e-4 lr_v=1e-3 c_e=0.02 batch_size=512 minibatch_size=128 epoch=4 seed={seed}");
    println!("episode,total_steps,step,entropy,approx_kl,clip_ratio");
    let mut env = CartPole::new(seed, &device);
    let mut agent = PPOAgent::new(
        seed,
        0.99,
        0.98,
        0.2,
        0.02,
        4,
        ActorCriticConfig::separated(
            1e-4,
            RmsPropConfig::new().init(),
            1e-3,
            RmsPropConfig::new().init()
        ),
        SequentialActorCriticNetwork::new(
            MLPEncoder::new(vec![4, 128], Activation::Relu(burn::nn::Relu), &device),
            MLPEncoder::new(vec![4, 128], Activation::Relu(burn::nn::Relu), &device),
            CategoricalHead::new(128, 2, &device),
            LinearVHead::new(128, 1, &device)
        )
    );
    let mut buffer = RolloutBuffer::new();
    let mut tape = Tape::new(&mut env);
    let mut total_steps = 0;
    let mut log = Default::default();
    for i in 0..=4000 {
        
        let mut step = 0;
        tape.reset(&mut env);
        loop {
            let action = agent.action(tape.obs.clone(), tape.constraint.clone());
            let dist = agent.dist(tape.obs.clone(), tape.constraint.clone());
            let (t, _, terminated, truncated) = tape.step(&mut env, action.clone());
            let t = t.add_extra(dist.log_probs(action));

            buffer.push(t);

            step += 1;
            total_steps += 1;
            if buffer.len() >= 512 {
                let batch = buffer.pop();
                (agent, log) = agent.update(128, batch);
            }
            if terminated || truncated { break; }
        }

        if i % 10 == 0 {
            let entropy = log.entropy();
            let approx_kl = log.approx_kl();
            let clip_ratio = log.clip_ratio();
            println!("{i},{total_steps},{step},{entropy},{approx_kl},{clip_ratio}");
            if i % 100 == 0 {
                eprintln!("Episode: {i}, Total steps: {total_steps} Steps: {step}, Entropy: {entropy}, Approx KL: {approx_kl} Clip ratio: {clip_ratio}");
            }
        }
    }
}