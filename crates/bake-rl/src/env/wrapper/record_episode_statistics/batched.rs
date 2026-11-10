//! Episode statistics recording wrapper
use burn::prelude::*;
use crate::core::env::BatchedEnvironment;
use crate::core::env::space::BatchedSpace;
use super::EpisodeStatistics;

/// records the episode-related statistics of batched environments, for scalar rewards
/// # What it records?
/// - Just as Gymnasium's `RecordEpisodeStatistics`, it records reward, step, wall-clock time and done masks, through the `Info`
pub struct BatchedRecordEpisodeStatistics<E: BatchedEnvironment<Reward = Tensor<1>>> {
    env: E,
    record_reward: Tensor<1>,
    record_step: Tensor<1, Int>,
}

impl<E: BatchedEnvironment<Reward = Tensor<1>>> BatchedRecordEpisodeStatistics<E> {
    /// create new `RecordEpisodeStatistics` wrapper, wrapping the given batched environment
    pub fn new(env: E) -> Self {
        let n = env.n_envs();
        let device = env.device();
        Self {
            env,
            record_reward: Tensor::from_floats(vec![0.; n].as_slice(), &device),
            record_step: Tensor::from_ints(vec![0; n].as_slice(), &device)
        }
    }
}

impl<E: BatchedEnvironment<Reward = Tensor<1>>> BatchedEnvironment for BatchedRecordEpisodeStatistics<E> {
    type ObsSpace = E::ObsSpace;
    type ActionSpace = E::ActionSpace;
    type Constraint = E::Constraint;
    type Reward = E::Reward;
    type Info = (E::Info, Vec<EpisodeStatistics>);

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
                self.record_step.inplace(|s| s.mask_fill(mask.clone(), 0));
            },
            None => {
                self.record_reward = Tensor::zeros_like(&self.record_reward);
                self.record_step = Tensor::zeros_like(&self.record_step);
            }
        }
        let ((o, c), i) = self.env.reset(mask.into());
        ((o, c), (i, vec![]))
    }

    fn step(&mut self, action: <Self::ActionSpace as BatchedSpace>::Element) -> ((<Self::ObsSpace as BatchedSpace>::Element, Self::Constraint), Self::Reward, Tensor<1, Bool>, Tensor<1, Bool>, Self::Info) {
        let ((o, c), r, ter, tru, i) = self.env.step(action);
        self.record_reward.inplace(|s| s + r.clone());
        self.record_step.inplace(|s| s + 1);
        let record_done = ter.clone().bool_or(tru.clone());

        let mut episode_stats = vec![];

        if record_done.clone().any().into_scalar() {
            let rew_vec = self.record_reward.clone().try_into_vec_as::<f32>().unwrap();
            let step_vec = self.record_step.clone().try_into_vec_as::<i64>().unwrap();
            let done_vec = record_done.try_into_vec_as::<bool>().unwrap();

            for (i, ((r, s), d)) in rew_vec.into_iter().zip(step_vec.into_iter()).zip(done_vec.into_iter()).enumerate() {
                if d {
                    episode_stats.push(EpisodeStatistics {
                        id: i,
                        r, s
                    });
                }
            }
        }

        ((o, c), r, ter, tru, (i, episode_stats))
    }
}