//! Episode statistics recording wrapper
use crate::core::env::Environment;
use crate::core::env::space::NativeSpace;
/// records the episode-related statistics of vector environments, for scalar rewards
/// # What it records?
/// - Just as Gymnasium's `RecordEpisodeStatistics`, it records reward, step, wall-clock time and done masks, through the `Info`
pub struct RecordEpisodeStatistics<E: Environment<Reward = f32>> {
    env: E,
    record_reward: f32,
    record_step: usize,
}

/// episode statistics
/// - r: accumulated rewards
/// - s: accumulated steps
#[derive(Debug, Clone)]
pub struct EpisodeStatistics {
    /// accumulated rewards
    pub r: f32,
    /// accumulated steps
    pub s: usize,
}

impl<E: Environment<Reward = f32>> RecordEpisodeStatistics<E> {
    /// create new `RecordEpisodeStatistics` wrapper, wrapping the given vector environment
    pub fn new(env: E) -> Self {
        Self {
            env,
            record_reward: 0.,
            record_step: 0,
        }
    }
}

impl<E: Environment<Reward = f32>> Environment for RecordEpisodeStatistics<E> {
    type ObsSpace = E::ObsSpace;
    type ActionSpace = E::ActionSpace;
    type Constraint = E::Constraint;
    type Reward = E::Reward;
    type Info = (E::Info, Option<EpisodeStatistics>);

    fn action_space(&self) -> &Self::ActionSpace {
        self.env.action_space()
    }

    fn obs_space(&self) -> &Self::ObsSpace {
        self.env.obs_space()
    }

    fn reset(&mut self) -> ((<Self::ObsSpace as NativeSpace>::Element, Self::Constraint), Self::Info) {
        let ((o, c), i) = self.env.reset();
        self.record_reward = 0.;
        self.record_step = 0;
        ((o, c), (i, None))
    }

    fn step(&mut self, action: <Self::ActionSpace as NativeSpace>::Element) -> ((<Self::ObsSpace as NativeSpace>::Element, Self::Constraint), Self::Reward, bool, bool, Self::Info) {
        let ((o, c), r, ter, tru, i) = self.env.step(action);
        self.record_reward += r;
        self.record_step += 1;
        let done = ter || tru;

        let mut episode_stat = None;

        if done {
            episode_stat = Some(EpisodeStatistics {
                r: self.record_reward,
                s: self.record_step
            })
        }

        ((o, c), r, ter, tru, (i, episode_stat))
    }
}