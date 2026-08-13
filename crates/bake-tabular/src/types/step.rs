pub struct Step {
    pub obs: usize,
    pub reward: f32,
    pub done: bool,
    pub truncated: bool,
}