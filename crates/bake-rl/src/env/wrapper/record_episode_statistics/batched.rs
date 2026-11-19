//! Episode statistics recording wrapper
use std::time::Instant;
use std::collections::VecDeque;
use burn::prelude::*;
use crate::core::env::BatchedEnvironment;
use crate::core::env::space::BatchedSpace;

/// records the episode-related statistics of batched environments, for scalar rewards
/// # What it records?
/// - Just as Gymnasium's `RecordEpisodeStatistics`, it records reward, step, wall-clock time (in microseconds) and done masks, through the `Info`
pub struct BatchedRecordEpisodeStatistics<E: BatchedEnvironment<Reward = Tensor<1>>> {
    env: E,
    record_reward: Tensor<1>,
    record_length: Tensor<1, Int>,
    record_clock: Tensor<1>,

    wrapper_start: Instant,

    buffer_size: usize,

    reward_buffer: VecDeque<Tensor<1>>,
    length_buffer: VecDeque<Tensor<1, Int>>,
    time_buffer: VecDeque<Tensor<1>>,
}

impl<E: BatchedEnvironment<Reward = Tensor<1>>> BatchedRecordEpisodeStatistics<E> {
    /// create new `RecordEpisodeStatistics` wrapper, wrapping the given batched environment
    pub fn new(env: E, buffer_size: usize) -> Self {
        let n = env.n_envs();
        let device = env.device();
        Self {
            env,
            record_reward: Tensor::from_floats(vec![0.; n].as_slice(), &device),
            record_length: Tensor::from_ints(vec![0; n].as_slice(), &device),
            record_clock: Tensor::from_floats(vec![0; n].as_slice(), &device),

            wrapper_start: Instant::now(),

            buffer_size,

            reward_buffer: VecDeque::with_capacity(buffer_size),
            length_buffer: VecDeque::with_capacity(buffer_size),
            time_buffer: VecDeque::with_capacity(buffer_size),
        }
    }
    /// return the mean of latest `self.buffer_size` (less if not fulfilled) episode rewards
    pub fn reward_mean(&self) -> Tensor<1> {
        let zero = Tensor::from_floats([0.], &self.device());
        self.reward_buffer.iter().cloned().fold(zero, |acc, element| acc + element) / self.reward_buffer.len() as f32
    }

    /// return the mean of latest `self.buffer_size` (less if not fulfilled) episode lengths
    pub fn length_mean(&self) -> Tensor<1> {
        let zero = Tensor::from_ints([0], &self.device());
        self.length_buffer.iter().cloned().fold(zero, |acc, element| acc + element).float() / self.length_buffer.len() as f32
    }

    /// return the mean of latest `self.buffer_size` (less if not fulfilled) episodes' elapsed time
    pub fn time_mean(&self) -> Tensor<1> {
        let zero = Tensor::from_floats([0], &self.device());
        self.time_buffer.iter().cloned().fold(zero, |acc, element| acc + element) / self.time_buffer.len() as f32
    }
}

impl<E: BatchedEnvironment<Reward = Tensor<1>>> BatchedEnvironment for BatchedRecordEpisodeStatistics<E> {
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

    fn device(&self) -> Device {
        self.env.device()
    }

    fn n_envs(&self) -> usize {
        self.env.n_envs()
    }

    fn reset(&mut self, mask: Option<Tensor<1, Bool>>) -> ((<Self::ObsSpace as BatchedSpace>::Element, Self::Constraint), Self::Info) {
        match mask.clone() {
            Some(mask) => {
                self.record_reward.inplace(|r| r.mask_fill(mask.clone(), 0.));
                self.record_length.inplace(|s| s.mask_fill(mask.clone(), 0));
                self.record_clock.inplace(|t| t.mask_fill(mask.clone(), (Instant::now() - self.wrapper_start).as_micros() as f32));
            },
            None => {
                self.record_reward = Tensor::zeros_like(&self.record_reward);
                self.record_length = Tensor::zeros_like(&self.record_length);
                self.record_clock = Tensor::full_like(&self.record_clock, (Instant::now() - self.wrapper_start).as_micros() as f32);
            }
        }
        let ((o, c), i) = self.env.reset(mask.into());
        ((o, c), (i, None))
    }

    fn step(&mut self, action: <Self::ActionSpace as BatchedSpace>::Element) -> ((<Self::ObsSpace as BatchedSpace>::Element, Self::Constraint), Self::Reward, Tensor<1, Bool>, Tensor<1, Bool>, Self::Info) {
        let ((o, c), r, ter, tru, i) = self.env.step(action);
        self.record_reward.inplace(|s| s + r.clone());
        self.record_length.inplace(|s| s + 1);

        let record_done = ter.clone().bool_or(tru.clone());
        let record_time = (Instant::now() - self.wrapper_start).as_micros() as f32 - self.record_clock.clone();
        let episode_stats = EpisodeStatistics {
            r: self.record_reward.clone(),
            l: self.record_length.clone(),
            t: record_time.clone(),
            m: record_done.clone(),
        };

        // break down
        let rewards = self.record_reward.clone().mask_select(record_done.clone()).split(1, 0);
        let lengths = self.record_length.clone().mask_select(record_done.clone()).split(1, 0);
        let times = record_time.mask_select(record_done).split(1, 0);

        for (r, (l, t)) in rewards.into_iter().zip(lengths.into_iter().zip(times.into_iter())) {
            if self.reward_buffer.len() == self.buffer_size { self.reward_buffer.pop_front(); }
            if self.length_buffer.len() == self.buffer_size { self.length_buffer.pop_front(); }
            if self.time_buffer.len() == self.buffer_size { self.time_buffer.pop_front(); }
            self.reward_buffer.push_back(r);
            self.length_buffer.push_back(l);
            self.time_buffer.push_back(t);
        }

        ((o, c), r, ter, tru, (i, Some(episode_stats)))
    }
}

/// episode statistics
/// - r: accumulated rewards
/// - l: accumulated length
/// - t: accumulated time in micros
/// - m: mask indicating which environment has ended (`true`: ended)
#[derive(Debug, Clone)]
pub struct EpisodeStatistics {
    /// accumulated rewards
    pub r: Tensor<1>,
    /// accumulated length
    pub l: Tensor<1, Int>,
    /// accumulated time in micros
    pub t: Tensor<1>,
    /// mask indicating which environment has ended (`true`: ended)
    pub m: Tensor<1, Bool>
}
