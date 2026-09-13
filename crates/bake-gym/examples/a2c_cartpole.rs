use bake_deep::distribution::Distribution;
use bake_deep::logger::MovingAvgLogger;
use bake_deep::{
    algorithm::{a2c::A2C, advantage_estimator::AdvantageEstimator},
    loss::Loss,
    buffer::RolloutBuffer,
    contract::ActorCritic,
    distribution::Categorical,
    env::Tape,
    net::basic::MlpSeparatedActorCriticNet,
    wrapper::ActorCriticWrapper
};
use bake_gym::env::{GymCartPole, KwArgs};

use burn::{nn::activation::ActivationConfig::Relu, optim::RmsPropConfig, tensor::Device};

pub fn main() {
    let seed: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(12);
    let device = Device::default();
    device.seed(seed);
    let autodiff_device = device.clone().autodiff();
    
    let state = A2C { gamma: 0.99, advantage: AdvantageEstimator::Gae { lambda: 0.95 }, loss_fn: Loss::MseLoss };
    let env = GymCartPole::new(seed, &device, KwArgs::new());
    let mut actor_critic: ActorCriticWrapper<_, Categorical> = ActorCriticWrapper::new(MlpSeparatedActorCriticNet::new(&[env.obs_shape()[1], 128, env.n_actions()], Relu, &autodiff_device));

    let lr_a = 1e-4;
    let lr_c = 1e-3;
    let mut opt_a = RmsPropConfig::new().init();
    let mut opt_c = RmsPropConfig::new().init();

    let mut buffer = RolloutBuffer::new();
    let mut tape = Tape::new(env);

    let mut logger = MovingAvgLogger::new();
    logger.register("reward", 100);
    logger.register("step", 100);
    logger.register("actor_loss", 100);
    logger.register("critic_loss", 100);
    logger.register("entropy", 100);

    for count in 0..=600000 {
        let action = actor_critic.action(tape.obs.clone(), tape.constraint.clone());
        let t = tape.step(action);
        buffer.push(t);

        if buffer.len() >= 128 {
            let batch = buffer.pop();
            let (net, loss) = A2C::loss(&state, actor_critic, batch);
            logger.push(&loss);
            actor_critic = A2C::update_separated(net, loss, 0.02, lr_a, &mut opt_a, lr_c, &mut opt_c)
        }

        if tape.done() {
            logger.push_single("reward", tape.episode_reward);
            logger.push_single("step", tape.steps as f32);
            tape.reset();
        }

        if count % 5000 == 0 {
            let ep_reward_average = logger.emit("reward");
            let actor_loss = logger.emit("actor_loss");
            let critic_loss = logger.emit("critic_loss");
            let entropy = logger.emit("entropy");
            println!("count: {count}, reward_avg: {ep_reward_average}, actor_loss: {actor_loss}, critic_loss: {critic_loss}, entropy: {entropy}");
        }
    }

    // evaluation
    let env = GymCartPole::new(seed, &device, KwArgs::new().add("render_mode", "human"));
    let mut tape = Tape::new(env);
    for _ in 0..5 {
        tape.reset();
        loop {
            let action = actor_critic.dist(tape.obs.clone(), tape.constraint.clone()).mode();
            tape.step(action);

            if tape.done() {
                break;
            }
        }
    }
}