//! Episode statistics recording wrapper
use std::collections::VecDeque;
use std::time::Instant;
use crate::core::env::Environment;
use crate::core::env::space::Space;

/// records the episode-related statistics of environments (for scalar rewards)
/// # What it records?
/// - Just as Gymnasium's `RecordEpisodeStatistics`, it records reward, step and wall-clock time (seconds) through the `Info`
pub struct RecordEpisodeStatistics<E: Environment<Reward = f32>> {
    env: E,
    record_reward: f32,
    record_length: i64,
    record_time: Instant,
    
    buffer_size: usize,
    reward_buffer: VecDeque<f32>,
    length_buffer: VecDeque<i64>,
    time_buffer: VecDeque<f32>,
}

impl<E: Environment<Reward = f32>> RecordEpisodeStatistics<E> {
    /// create new `RecordEpisodeStatistics` wrapper, wrapping the given environment
    pub fn new(env: E, buffer_size: usize) -> Self {
        
        Self {
            env,
            record_reward: 0.,
            record_length: 0,
            record_time: Instant::now(),

            buffer_size,
            reward_buffer: VecDeque::with_capacity(buffer_size),
            length_buffer: VecDeque::with_capacity(buffer_size),
            time_buffer: VecDeque::with_capacity(buffer_size),
        }
    }

    /// return the mean of latest `self.buffer_size` (less if not fulfilled) episode rewards
    pub fn reward_mean(&self) -> f32 {
        self.reward_buffer.iter().sum::<f32>() / self.reward_buffer.len() as f32
    }

    /// return the mean of latest `self.buffer_size` (less if not fulfilled) episode lengths
    pub fn length_mean(&self) -> f32 {
        self.length_buffer.iter().sum::<i64>() as f32 / self.length_buffer.len() as f32
    }

    /// return the mean of latest `self.buffer_size` (less if not fulfilled) episodes' elapsed time
    pub fn time_mean(&self) -> f32 {
        self.time_buffer.iter().sum::<f32>() / self.time_buffer.len() as f32
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

    fn reset(&mut self) -> ((<Self::ObsSpace as Space>::Element, Self::Constraint), Self::Info) {
        let ((o, c), i) = self.env.reset();
        self.record_reward = 0.;
        self.record_length = 0;
        self.record_time = Instant::now();

        ((o, c), (i, None))
    }

    fn step(&mut self, action: <Self::ActionSpace as Space>::Element) -> ((<Self::ObsSpace as Space>::Element, Self::Constraint), Self::Reward, bool, bool, Self::Info) {
        let ((o, c), r, ter, tru, i) = self.env.step(action);
        self.record_reward += r;
        self.record_length += 1;
        let done = ter || tru;

        let mut episode_stat = None;

        if done {
            let elapsed = self.record_time.elapsed().as_secs_f32();
            episode_stat = Some(EpisodeStatistics {
                r: self.record_reward,
                l: self.record_length,
                t: elapsed
            });
            if self.reward_buffer.len() == self.buffer_size { self.reward_buffer.pop_front(); }
            if self.length_buffer.len() == self.buffer_size { self.length_buffer.pop_front(); }
            if self.time_buffer.len() == self.buffer_size { self.time_buffer.pop_front(); }

            self.reward_buffer.push_back(self.record_reward);
            self.length_buffer.push_back(self.record_length);
            self.time_buffer.push_back(elapsed);
        }

        ((o, c), r, ter, tru, (i, episode_stat))
    }
}

/// episode statistics
/// - r: accumulated rewards
/// - l: accumulated length
/// - t: accumulated time in secs
#[derive(Debug, Clone)]
pub struct EpisodeStatistics {
    /// accumulated rewards
    pub r: f32,
    /// accumulated length
    pub l: i64,
    /// accumulated time in secs
    pub t: f32,
}
