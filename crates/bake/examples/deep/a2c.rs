use bake_deep::{algorithm::*, approximator::{ActorCritic, SeparatedActorCritic, encoder::MLPEncoder, head::{CategoricalHead, LinearVHead}}, buffer::RolloutBuffer, env::CartPole, types::Tape};
use burn::{nn::activation::Activation, optim::RmsPropConfig, tensor::Device};


pub fn main() {
    let seed: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(12);
    let device = Device::default().autodiff();
    device.seed(seed);
    println!("# optimizer=rmsprop lr_p=1e-4 lr_v=1e-3 c_e=0.02 T=160 seed={seed}");
    println!("episode,total_steps,step,entropy,value_loss");
    let mut env = CartPole::new(seed, &device);
    let alg = A2C::new(0.99, 0.95, dqn::ValueLoss::MseLoss);
    let mut actor_critic = SeparatedActorCritic::new(
            MLPEncoder::new(vec![4, 128], Activation::Relu(burn::nn::Relu), &device),
            MLPEncoder::new(vec![4, 128], Activation::Relu(burn::nn::Relu), &device),
            CategoricalHead::new(128, 2, &device),
            LinearVHead::new(128, 1, &device)
        );
    let c_e = 0.02;
    let mut opt_a = RmsPropConfig::new().init();
    let mut opt_c = RmsPropConfig::new().init();
    let lr_a = 1e-4;
    let lr_c = 1e-3;

    let mut buffer = RolloutBuffer::new();
    let mut tape = Tape::new(&mut env);
    let mut total_steps = 0;

    for i in 0..=4000 {
        let mut step = 0;
        tape.reset(&mut env);
        loop {
            let action = actor_critic.action(tape.obs.clone(), tape.constraint.clone());
            let t = tape.step(&mut env, action);
            buffer.push(t);

            step += 1;
            total_steps += 1;
            if buffer.len() >= 160 {
                let batch = buffer.pop();
                let loss = A2C::loss(&alg, &actor_critic, batch);
                actor_critic = A2C::update_separated(actor_critic, loss, c_e, lr_a, &mut opt_a, lr_c, &mut opt_c)
            }
            if tape.done() { break; }
        }

        if i % 10 == 0 {
            let loss: f32 = log["value_loss"].into_scalar();
            let entropy: f32 = log["entropy"].into_scalar();
            println!("{i},{total_steps},{step},{entropy},{loss}");
            if i % 100 == 0 {
                eprintln!("Episode: {i}, Total steps: {total_steps} Steps: {step}, Entropy: {entropy}, Loss: {loss}");
            }
        }
    }
}