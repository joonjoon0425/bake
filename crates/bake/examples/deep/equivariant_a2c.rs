use bake_deep::{algorithm::*, approximator::{ActorCritic, Encoder, Head, VHead, encoder::MlpEncoder, head::{CategoricalHead, LinearVHead}}, buffer::RolloutBuffer, constraint::DiscreteConstraint, distribution::Categorical, env::CartPole, types::{Logger, Tape}};
use burn::{Tensor, module::Module, nn::activation::ActivationConfig::Relu, optim::RmsPropConfig, tensor::Device};


pub fn main() {
    let seed: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(12);
    let device = Device::default().autodiff();
    device.seed(seed);
    let mut env = CartPole::new(seed, &device);
    let config = A2C::new(0.99, 0.95, dqn::ValueLoss::MseLoss);
    let mut actor_critic = Z2Symmetrized::new(
            MlpEncoder::new(vec![4, 128], Relu, &device),
            MlpEncoder::new(vec![4, 128], Relu, &device),
            CategoricalHead::new(128, 2, &device),
            LinearVHead::new(128, &device)
        );
    let c_e = 0.02;
    let mut opt_a = RmsPropConfig::new().init();
    let mut opt_c = RmsPropConfig::new().init();
    let lr_a = 1e-4;
    let lr_c = 1e-3;

    let mut buffer = RolloutBuffer::new();
    let mut tape = Tape::new(&mut env);
    let mut total_steps = 0;
    let mut logger = Logger::default();

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
                let loss = A2C::loss(&config, &actor_critic, batch);
                logger.record(&loss);
                actor_critic = A2C::update_separated(actor_critic, loss, c_e, lr_a, &mut opt_a, lr_c, &mut opt_c)
            }
            if tape.done() { break; }
        }

        if i % 10 == 0 {
            let mean = logger.mean();
            let loss = mean.get("critic_loss").unwrap_or(&0f32);
            let entropy = mean.get("entropy").unwrap_or(&0f32);
            println!("{i},{total_steps},{step},{entropy},{loss}");
            if i % 100 == 0 {
                eprintln!("Episode: {i}, Total steps: {total_steps} Steps: {step}, Entropy: {entropy}, Loss: {loss}");
            }
            logger.clear()
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

    fn dist(&self, obs: Self::Obs, constraint: Self::Constraint) -> Self::Dist {
        let dist_pos = self.policy_head.forward(self.policy_encoder.forward(obs.clone()), constraint.clone());
        let dist_neg = self.policy_head.forward(self.policy_encoder.forward(-(obs.clone())), constraint.clone());
        let logits = (dist_pos.logits().clone() + dist_neg.logits().clone().flip([1])) * 0.5;
        Categorical::new(logits, constraint)
    }

    fn value(&self, obs: Self::Obs) -> Tensor<1> {
        let v_pos = self.value_head.forward(self.value_encoder.forward(obs.clone()));
        let v_neg = self.value_head.forward(self.value_encoder.forward(-obs));
        let value = (v_pos + v_neg) * 0.5;
        value
    }

    fn shares_encoder(&self) -> bool { false }
}