pub struct QLearningAgent {
    gamma: f32,
    alpha: f32,

    q_table: Vec<f32>,

    n_states: usize,
    n_actions: usize,
}

impl QLearningAgent {
    pub fn new(n_states: usize, n_actions: usize, alpha: f32, gamma: f32) -> Self {
        Self {
            gamma,
            alpha,
            q_table: vec![0f32; n_states * n_actions],
            n_actions,
            n_states,
        }
    }

    pub fn action(policy: Policy, )
}