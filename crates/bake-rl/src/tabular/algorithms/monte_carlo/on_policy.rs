use crate::core::constraint::DiscreteConstraint;
use crate::core::data::Transition;
use crate::tabular::algorithms::monte_carlo::MonteCarloStat;
use crate::tabular::qtable::QTable;
use super::MonteCarloVisit;

/// state struct for on-policy Monte Carlo
#[derive(Debug, Clone)]
pub struct OnPolicyMonteCarloState {
    /// discount rate
    pub gamma: f32,
    /// select one of two methods: first-visit Monte Carlo and every-visit Monte Carlo
    pub visit: MonteCarloVisit,
}

/// update q table using on-policy Monte Carlo algorithm
pub fn on_policy_monte_carlo_update(state: &OnPolicyMonteCarloState, stat: &mut MonteCarloStat, qtable: &mut QTable, episode: Vec<Transition<usize, impl DiscreteConstraint, usize, f32>>) {
    let n_obs = qtable.n_obs();
    let n_action = qtable.n_action();
    assert!(qtable.n_obs() * qtable.n_action() == stat.visit_count.len(), "qtable.n_obs * qtable.n_action != stat.visit_count's table size");

    // algorithm
    let mut goal = 0f32;

    let first_visit_idx = if matches!(state.visit, MonteCarloVisit::FirstVisit) {
        let mut idx = vec![-1; n_obs * n_action];
        for (i, t) in episode.iter().enumerate() {
            if idx[t.obs * n_action + t.action] == -1 { idx[t.obs * n_action + t.action] = i as i32 }
        }
        idx
    } else { vec![] };

    for (i, t) in episode.into_iter().enumerate().rev() {
        goal = state.gamma * goal + t.reward;
        let flat_idx = t.obs * n_action + t.action;
        let cur_q = qtable.qvalue(t.obs, t.action);
        match state.visit {
            MonteCarloVisit::FirstVisit => {
                if first_visit_idx[flat_idx] == i as i32 {
                    stat.visit_count[flat_idx] += 1;
                    qtable.update(t.obs, t.action, cur_q + (goal - cur_q) / stat.visit_count[flat_idx] as f32);
                }
            },
            MonteCarloVisit::EveryVisit => {
                stat.visit_count[flat_idx] += 1;
                qtable.update(t.obs, t.action, cur_q + (goal - cur_q) / stat.visit_count[flat_idx] as f32);
            }
        }
    }
}