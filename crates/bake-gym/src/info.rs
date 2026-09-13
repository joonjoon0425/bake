//! Traits of informations of Gymnasium environment

/// Information of tabular (toy text) Gymnasium environment
/// - Uses one-hot encoding
pub trait TabularGymEnvInfo {
    /// the number of observations
    fn n_obs() -> usize;
    /// the number of actions
    fn n_actions() -> usize;
    /// the name of the environment
    fn name() -> &'static str;
}

/// Information of discrete action Gymnasium environment
pub trait DiscreteGymEnvInfo<const O: usize> {
    /// the shape of observations
    fn obs_shape() -> [usize; O];
    /// the number of actions
    fn n_actions() -> usize;
    /// the name of the environment
    fn name() -> &'static str;
}

/// Information of continuous action Gymnasium environment
pub trait ContinuousGymEnvInfo<const O: usize, const A: usize> {
    /// the dimension of observations
    fn obs_dim() -> usize;
    /// the dimension of actions
    fn action_dim() -> usize;
    /// the name of environment
    fn name() -> &'static str;
}