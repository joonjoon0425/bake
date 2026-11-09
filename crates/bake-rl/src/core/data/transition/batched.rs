use crate::core::data::BatchOps;
use crate::core::data::dict::BatchedDict;
use burn::prelude::*;

/// transition struct
pub struct BatchedTransition<Obs, Action, Constraint, Reward>
where 
    Obs: BatchOps,
    Action: BatchOps,
    Constraint: BatchOps,
    Reward: BatchOps
{
    obs: Obs,
    action: Action,
    reward: Reward,
    next_obs: Obs,

    constraint: Constraint,
    next_constraint: Constraint,

    terminated: Tensor<1, Bool>,
    truncated: Tensor<1, Bool>,

    dict: BatchedDict
}

impl<Obs, Action, Constraint, Reward> BatchedTransition<Obs, Action, Constraint, Reward>
where 
    Obs: BatchOps,
    Action: BatchOps,
    Constraint: BatchOps,
    Reward: BatchOps
{
    /// create new transition
    pub fn new(
        obs: Obs,
        action: Action,
        reward: Reward,
        next_obs: Obs,
        constraint: Constraint,
        next_constraint: Constraint,
        terminated: Tensor<1, Bool>,
        truncated: Tensor<1, Bool>,
        dict: BatchedDict,
    ) -> Self {
        Self {
            obs, action, reward, next_obs, constraint, next_constraint, terminated, truncated, dict
        }
    }

    pub fn obs(&self) -> &Obs { &self.obs }
    pub fn action(&self) -> &Action { &self.action }
    pub fn reward(&self) -> &Reward { &self.reward }
    pub fn next_obs(&self) -> &Obs { &self.next_obs }
    pub fn constriant(&self) -> &Constraint { &self.constraint }
    pub fn next_constraint(&self) -> &Constraint { &self.next_constraint }
    pub fn terminated(&self) -> &Tensor<1, Bool> { &self.terminated }
    pub fn truncated(&self) -> &Tensor<1, Bool> { &self.truncated }
    pub fn dict(&self) -> &BatchedDict { &self.dict }
    pub fn dict_mut(&mut self) -> &mut BatchedDict { &mut self.dict }
}