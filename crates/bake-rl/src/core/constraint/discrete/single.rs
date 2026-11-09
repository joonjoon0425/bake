/// discrete constraint for discrete action spaces
pub trait DiscreteConstraint: Clone {
    /// Fill the given `values` with `fill_value` where mask is false
    fn apply(&self, values: &[f32], fill_value: f32) -> Vec<f32>;
    /// return the possible number of actions
    /// - if `None` is returned, then all actions are possible
    fn n_possible_actions(&self) -> Option<usize>;
    /// returns if the given action is possible
    fn is_possible(&self, action: usize) -> bool;
}

impl DiscreteConstraint for Vec<bool> {
    fn apply(&self, values: &[f32], fill_value: f32) -> Vec<f32> {
        assert!(self.len() == values.len(), "DiscreteConstraint::apply: action number mismatch, constraint: {}, value: {}", self.len(), values.len());
        let mut masked = vec![0f32; values.len()];
        for (idx, (&b, &value)) in self.iter().zip(values.iter()).enumerate() {
            if !b {
                masked[idx] = fill_value;
            } else {
                masked[idx] = value;
            }
        }
        masked
    }

    fn is_possible(&self, action: usize) -> bool {
        self[action]
    }

    fn n_possible_actions(&self) -> Option<usize> {
        Some(self.iter().filter(|b| **b).count())
    }
}

impl<const N: usize> DiscreteConstraint for [bool; N] {
    fn apply(&self, values: &[f32], fill_value: f32) -> Vec<f32> {
        assert!(self.len() == values.len(), "DiscreteConstraint::apply: action number mismatch, constraint: {}, value: {}", self.len(), values.len());
        let mut masked = vec![0f32; values.len()];
        for (idx, (&b, &value)) in self.iter().zip(values.iter()).enumerate() {
            if !b {
                masked[idx] = fill_value;
            } else {
                masked[idx] = value;
            }
        }
        masked
    }

    fn is_possible(&self, action: usize) -> bool {
        self[action]
    }

    fn n_possible_actions(&self) -> Option<usize> {
        Some(self.iter().filter(|b| **b).count())
    }
}

impl DiscreteConstraint for () {
    fn apply(&self, values: &[f32], _fill_value: f32) -> Vec<f32> {
        values.iter().copied().collect()
    }

    fn is_possible(&self, _action: usize) -> bool {
        true
    }

    fn n_possible_actions(&self) -> Option<usize> {
        None
    }
}