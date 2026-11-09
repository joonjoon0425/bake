
/// transition struct
pub struct Transition<Obs, Action, Reward, Constraint> {
    obs: Obs,
    action: Action,
    reward: Reward,
    next_obs: Obs,

    constraint: Constraint,
    next_constraint: Constraint,

    terminated: bool,
    truncated: bool,
}