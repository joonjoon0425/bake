use bake_deep::{algorithm::*, approximator::{ActorCritic, CategoricalActorCritic}, buffer::RolloutBuffer, env::CartPole, network::{ActorCriticNet, EncoderType::Separated, MlpActorCriticNet}, types::{Logger, Tape}};
use burn::{Tensor, module::Module, nn::activation::ActivationConfig::Relu, optim::RmsPropConfig, tensor::Device};


pub fn main() {
    let seed: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(12);
    let device = Device::default().autodiff();
    device.seed(seed);
    let mut env = CartPole::new(seed, &device);
    let config = A2C::new(0.99, 0.95, dqn::ValueLoss::MseLoss);
    let mut actor_critic = CategoricalActorCritic::new(Z2SymNet::new(MlpActorCriticNet::new(&[4, 128, 2], Relu, &device)));
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
pub struct Z2SymNet<Ac: ActorCriticNet> {
    net: Ac,
}

impl<Ac: ActorCriticNet> Z2SymNet<Ac> {
    pub fn new(net: Ac) -> Self {
        Self {
            net
        }
    }
}

impl<Ac: ActorCriticNet<Obs = Tensor<2>, Params = Tensor<2>>> ActorCriticNet for Z2SymNet<Ac> {
    type Obs = Tensor<2>;
    type Params = Tensor<2>;

    fn params(&self, obs: Self::Obs) -> Self::Params {
        let logits_pos = self.net.params(obs.clone());
        let logits_neg = self.net.params(-obs);
        (logits_pos + logits_neg.flip([1])) * 0.5
    }

    fn values(&self, obs: Self::Obs) -> Tensor<1> {
        let v_pos = self.net.values(obs.clone());
        let v_neg = self.net.values(-obs);
        (v_pos + v_neg) * 0.5
    }

    fn encoder_type(&self) -> bake_deep::network::EncoderType {
        Separated
    }
}