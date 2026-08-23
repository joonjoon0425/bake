use bake_deep::{agent::A2CAgent, buffer::RolloutBuffer, config::ActorCriticConfig, constraint::{DiscreteConstraint}, distribution::Categorical, approximator::encoder::{Encoder, MLPEncoder}, env::CartPole, approximator::head::{CategoricalHead, Head, LinearVHead, VHead}, approximator::ActorCritic, types::Tape};
use burn::{Tensor, module::Module, nn::activation::Activation, optim::{RmsPropConfig}, tensor::Device};


pub fn main() {
    let seed: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(12);
    let device = Device::default().autodiff();
    device.seed(seed);
    println!("# optimizer=rmsprop lr_p=1e-4 lr_v=1e-3 c_e=0.02 T=160 seed={seed}");
    println!("episode,total_steps,step,entropy,value_loss");
    let mut env = CartPole::new(seed, &device);
    let mut agent = A2CAgent::new(
        0.99,
        0.95,
        0.02,
        ActorCriticConfig::separated(
            1e-4,
            RmsPropConfig::new().init(),
            1e-3,
            RmsPropConfig::new().init()
        ),
        Z2Symmetrized::new(
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
            let t = tape.step(&mut env, action);

            buffer.push(t);

            step += 1;
            total_steps += 1;
            if buffer.len() >= 160 {
                let batch = buffer.pop();
                (agent, log) = agent.update(batch);
            }
            if tape.done() { break; }
        }

        if i % 10 == 0 {
            let loss = log.value_loss();
            let entropy = log.entropy();
            // println!("{i},{total_steps},{step},{entropy},{loss}");
            if i % 100 == 0 {
                eprintln!("Episode: {i}, Total steps: {total_steps} Steps: {step}, Entropy: {entropy}, Loss: {loss}");
            }
        }
    }
}

#[derive(Module, Debug)]
pub struct Z2Symmetrized<E: Encoder, Ph: Head<Output = Categorical>, Vh: VHead> {
    policy_encoder: E,
    value_encoder: E,
    policy_head: Ph,
    value_head: Vh,
}

impl<E: Encoder, Ph: Head<Output = Categorical>, Vh: VHead> Z2Symmetrized<E, Ph, Vh> {
    pub fn new(policy_encoder: E, value_encoder: E, policy_head: Ph, value_head: Vh) -> Self {
        Self {
            policy_encoder,
            value_encoder,
            policy_head,
            value_head,
        }
    }
}

impl<E: Encoder<Obs = Tensor<2>>, Ph: Head<Output = Categorical, Constraint: DiscreteConstraint>, Vh: VHead> ActorCritic for Z2Symmetrized<E, Ph, Vh> {
    type Obs = E::Obs;
    type Constraint = <Ph as Head>::Constraint;
    type Dist = Ph::Output;

    fn actor(&self, obs: Self::Obs, constraint: Self::Constraint) -> Self::Dist {
        let dist_pos = self.policy_head.forward(self.policy_encoder.forward(obs.clone()), constraint.clone());
        let dist_neg = self.policy_head.forward(self.policy_encoder.forward(-(obs.clone())), constraint.clone());
        let logits = (dist_pos.logits().clone() + dist_neg.logits().clone().flip([1])) * 0.5;
        Categorical::new(logits, constraint)
    }

    fn critic(&self, obs: Self::Obs) -> Tensor<1> {
        let v_pos = self.value_head.forward(self.value_encoder.forward(obs.clone()));
        let v_neg = self.value_head.forward(self.value_encoder.forward(-obs));
        let value = (v_pos + v_neg) * 0.5;
        value
    }
}