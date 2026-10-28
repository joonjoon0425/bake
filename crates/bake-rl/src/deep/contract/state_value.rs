//! The state value trait

use crate::core::data::BatchTensor;
use crate::deep::network::Network;

/// The state value trait.
/// - The network which implements this trait is able to produce the state values
pub trait StateValue: Network {
    /// returns the state values of given observations
    fn state_value(&self, obs: Self::Obs) -> BatchTensor<1>;
}