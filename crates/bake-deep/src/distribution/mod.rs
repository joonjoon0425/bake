//! A probability distribution traits and structs
//! 
use burn::Tensor;
use crate::data::batchable::Batchable;

/// Probability distribution trait
pub trait Distribution : std::fmt::Debug + Clone + Sync + Send + 'static {
    /// the sample type which it produces
    type Sample: Batchable;
    /// the parmeters which distribution requires
    type Params;

    /// Sample from distribution
    fn sample(&self) -> Self::Sample;
    /// Get the most possible sample
    fn mode(&self) -> Self::Sample;

    /// get the log probabilities of given sample
    fn log_probs(&self, action: Self::Sample) -> Tensor<1>; // [batch]
    /// compute the entropies, for each batch dimensions
    fn entropy(&self) -> Tensor<1>; // [batch]
}

/// A trait which specifies which constraints the distributions can be applied to
pub trait PossibleConstraint<D: Distribution> {
    /// create a distribution from constraint and parameters
    fn create_distribution(params: D::Params, constraint: Self) -> D;
}

pub mod categorical;
pub use categorical::{Categorical};