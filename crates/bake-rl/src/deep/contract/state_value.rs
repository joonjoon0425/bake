//! The state value trait
use crate::deep::network::Network;
use burn::prelude::*;

/// The state value trait.
/// - The network which implements this trait is able to produce the state values
pub trait StateValue: Network {
    /// returns the state values of given observations
    fn state_value(&self, obs: Self::Obs) -> Tensor<1>;
}