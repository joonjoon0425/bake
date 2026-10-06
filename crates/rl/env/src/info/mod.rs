//! Information traits for environments
//! 
//! 
//! 

pub trait EnvironmentInfo {
    type Obs;
    type Action;
}

pub trait TabularEnvironmentInfo : EnvironmentInfo {
    fn n_obs() -> usize;
    fn n_actions() -> usize;
}

pub trait DiscreteEnvironmentInfo<const O: usize> : EnvironmentInfo {
    fn obs_shape() -> [usize; O];
    fn obs_range() -> (Self::Obs, Self::Obs);

    fn n_actions() -> usize;
}

pub trait ContinuousEnvironmentInfo<const O: usize, const A: usize> : EnvironmentInfo {
    fn obs_shape() -> [usize; O];
    fn obs_range() -> (Self::Obs, Self::Obs);
    
    fn action_shape() -> [usize; A];
    fn action_range() -> (Self::Action, Self::Action);
}