use bake::tabular::agent::*;
use bake::tabular::env::*;
use bake_tabular::policy::EpsGreedy;
use bake_tabular::types::Transition;

fn main() {
    let mut env = GridWorld::new();
    let mut agent = QLearningAgent::new(env.n_states(), env.n_actions(), 0.3, 0.99);
    let mut policy = EpsGreedy::new(2,1f32);

    for i in 0..100000 {
        let (mut obs, mut mask) = env.reset();
        let mut episode_reward = 0f32;
        let mut n_steps = 0usize;
        loop {
            let action = agent.action(&mut policy, obs, mask);
            let (next_obs, reward, terminated, truncated, next_mask) = env.step(action);
            agent.update(Transition {
                obs,
                action,
                reward,
                next_obs,
                terminated,
                truncated,
                mask,
                next_mask,
                extra: ()
            });

            obs = next_obs;
            mask = next_mask;
            
            episode_reward += reward;
            n_steps += 1;
            if terminated || truncated { break; }
        }
        *policy.eps_mut() *= 0.999;
        if i % 1000 == 0 { println!("Episode {i}, Steps: {n_steps}, Reward: {episode_reward}, Eps: {}", policy.eps()) }
        
    }
    
}