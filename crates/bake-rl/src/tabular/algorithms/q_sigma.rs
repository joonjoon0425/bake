//! Q(sigma)
//! 

/// state struct for Q(sigma)
#[derive(Debug, Clone)]
pub struct QSigmaState {
    /// discount rate
    pub gamma: f32,
    /// learning rate
    pub alpha: f32,
    /// ratio of tree-backup method (the opposite is per-decision with control variate)
    /// - currently supports only the constant ratio
    pub sigma: f32,
}