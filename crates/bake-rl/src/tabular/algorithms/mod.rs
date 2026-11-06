//! Implementations of tabular reinforcement learning algorithms

pub mod q_learning;

/// utility max function with index
fn argmax_max(values: &[f32]) -> (usize, f32) {
    let mut max = f32::MIN;
    let mut idx = 0;

    for (i, &v) in values.iter().enumerate() {
        if max < v {
            max = v;
            idx = i;
        }
    }
    (idx, max)
}