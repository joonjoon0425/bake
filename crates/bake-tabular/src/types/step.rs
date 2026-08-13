//! A Step which environment returns

/// A Step struct which environment returns
/// # WILL BE DELETED SOON
#[derive(Debug, Clone, Copy)]
pub struct Step {
    /// observation
    pub obs: usize,
    /// reward
    pub reward: f32,
    /// termianted
    pub done: bool,
    /// truncated
    pub truncated: bool,
}