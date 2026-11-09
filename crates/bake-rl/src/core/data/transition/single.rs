/// transition struct
pub struct Transition<Obs, Constraint, Action, Reward> {
    /// observation (time step `t`)
    pub obs: Obs,
    /// constraint of actions on observation (time step `t`)
    pub constraint: Constraint,
    /// action (time step `t + 1`)
    pub action: Action,
    /// reward (time step `t + 1`)
    pub reward: Reward,
    /// observation after the action (time step `t + 1`)
    pub next_obs: Obs,
    /// constraint of actions on observation after the action (time step `t + 1`)
    pub next_constraint: Constraint,
    
    /// true if next_obs (time step `t + 1`) is terminal state
    pub terminated: bool,
    /// true if next step (time step `t + 1`) is truncated
    pub truncated: bool,

    /// dictionary for additional elements
    pub dict: ()
}