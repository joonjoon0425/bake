use crate::core::constraint::DiscreteConstraint;
use crate::core::data::Transition;
use crate::tabular::algorithms::argmax_max;
use crate::tabular::qtable::QTable;

/// state struct for off-policy Monte Carlo
#[derive(Debug, Clone)]
pub struct OffPolicyMonteCarloState {
    /// discount rate
    pub gamma: f32,
}

/// struct for on policy Monte Carlo statistics for `update` (accumulated return and visit count, etc)
#[derive(Debug, Clone)]
pub struct OffPolicyMonteCarloStat {
    /// cummulative sum of weights of each pair (s, a)
    pub cumulative_weight: Vec<f32>,
}

impl OffPolicyMonteCarloStat {
    /// create new `MonteCarloStat`
    pub fn new(qtable: &QTable) -> Self {
        Self { cumulative_weight: vec![0f32; qtable.n_obs() * qtable.n_action()] }
    }
}

/// update q table using off-policy Monte Carlo algorithm
/// # Note
/// - requires following `key: element type` in transitions
///     - `action_prob: f32` for probability of action selection of behavior policy
pub fn off_policy_monte_carlo_update(state: &OffPolicyMonteCarloState, stat: &mut OffPolicyMonteCarloStat, qtable: &mut QTable, episode: Vec<Transition<usize, impl DiscreteConstraint, usize, f32>>) {
    let mut g = 0f32;
    let mut w = 1f32;

    for mut t in episode.into_iter().rev() {
        let flat_idx = t.obs * qtable.n_action() + t.action;
        let q_value = qtable.qvalue(t.obs, t.action);
        g = state.gamma * g + t.reward;
        stat.cumulative_weight[flat_idx] += w;
        qtable.update(t.obs, t.action, q_value + w / stat.cumulative_weight[flat_idx] * (g - q_value));

        let (greedy_action, _) = argmax_max(&qtable.qvalues(t.obs, t.constraint));
        if greedy_action != t.action { break; }
        w *= 1. / t.dict.remove_str::<f32>("action_prob").expect("`off_policy_monte_carlo_update` expects key `action_prob` with type `f32`")
    }
}