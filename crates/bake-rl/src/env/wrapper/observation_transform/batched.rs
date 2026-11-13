use crate::core::env::space_transform::BatchedSpaceTransform;
use crate::core::env::space::BatchedSpace;
use crate::core::env::BatchedEnvironment;

/// observation transform
pub struct BatchedObservationTransform<E: BatchedEnvironment, Transformer: BatchedSpaceTransform<E::ObsSpace>> {
    env: E,
    transformed_obs_space: Transformer::Applied,
    transformer: Transformer,
}

impl<E: BatchedEnvironment, Transformer: BatchedSpaceTransform<E::ObsSpace>> BatchedObservationTransform<E, Transformer> {
    /// create new `BatchedObservationTransform` with given environment and transformer
    pub fn new(env: E, transformer: Transformer) -> Self {
        let transformed_obs_space = transformer.transform_space(env.obs_space());
        Self {
            env,
            transformed_obs_space,
            transformer
        }
    }
}

impl<E: BatchedEnvironment, Transformer: BatchedSpaceTransform<E::ObsSpace>> BatchedEnvironment for BatchedObservationTransform<E, Transformer> {
    type ObsSpace = Transformer::Applied;
    type ActionSpace = E::ActionSpace;
    type Constraint = E::Constraint;
    type Reward = E::Reward;
    type Info = E::Info;

    fn obs_space(&self) -> &Self::ObsSpace {
        &self.transformed_obs_space
    }

    fn action_space(&self) -> &Self::ActionSpace {
        self.env.action_space()
    }

    fn n_envs(&self) -> usize {
        self.env.n_envs()
    }

    fn device(&self) -> burn::prelude::Device {
        self.env.device()
    }

    fn reset(&mut self, mask: Option<burn::prelude::Tensor<1, burn::prelude::Bool>>) -> ((<Self::ObsSpace as BatchedSpace>::Element, Self::Constraint), Self::Info) {
        let ((o, c), i) = self.env.reset(mask);
        let o = self.transformer.transform_element(self.env.obs_space(), o);
        ((o, c), i)
    }

    fn step(&mut self, action: <Self::ActionSpace as BatchedSpace>::Element) -> ((<Self::ObsSpace as BatchedSpace>::Element, Self::Constraint), Self::Reward, burn::prelude::Tensor<1, burn::prelude::Bool>, burn::prelude::Tensor<1, burn::prelude::Bool>, Self::Info) {
        let ((o, c), r, ter, tru, i) = self.env.step(action);
        let o = self.transformer.transform_element(self.env.obs_space(), o);
        ((o, c), r, ter, tru, i)
    }
}