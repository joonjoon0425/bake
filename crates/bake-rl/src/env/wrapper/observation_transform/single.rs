use crate::core::env::Environment;
use crate::core::env::space::Space;
use crate::core::env::space_transform::SpaceTransform;

/// observation transform
pub struct ObservationTransform<E: Environment, Transformer: SpaceTransform<E::ObsSpace>> {
    env: E,
    transformed_obs_space: Transformer::Applied,
    transformer: Transformer,
}

impl<E: Environment, Transformer: SpaceTransform<E::ObsSpace>> ObservationTransform<E, Transformer> {
    /// create new `ObservationTransform` with given environment and transformer
    pub fn new(env: E, transformer: Transformer) -> Self {
        let transformed_obs_space = transformer.transform_space(env.obs_space());
        Self {
            env,
            transformed_obs_space,
            transformer
        }
    }
}

impl<E: Environment, Transformer: SpaceTransform<E::ObsSpace>> Environment for ObservationTransform<E, Transformer> {
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

    fn reset(&mut self) -> ((<Self::ObsSpace as Space>::Element, Self::Constraint), Self::Info) {
        let ((o, c), i) = self.env.reset();
        let o = self.transformer.transform_element(self.env.obs_space(), o);
        ((o, c), i)
    }

    fn step(&mut self, action: <Self::ActionSpace as Space>::Element) -> ((<Self::ObsSpace as Space>::Element, Self::Constraint), Self::Reward, bool, bool, Self::Info) {
        let ((o, c), r, ter, tru, i) = self.env.step(action);
        let o = self.transformer.transform_element(self.env.obs_space(), o);
        ((o, c), r, ter, tru, i)
    }
}