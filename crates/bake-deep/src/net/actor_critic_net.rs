//! Network trait for Actor Critic methods
//! 
use burn::{module::{AutodiffModule, ModuleDisplay}, prelude::*};
use crate::{contract::actor_critic::EncoderType, data::batchable::Batchable};

/// Actor critic network for actor-critic methods
/// - the users must implement this trait to use their own network structure, and wrap it with wrapper.
/// # Warning
/// - the user must create the network initialy on autodiff device.
pub trait ActorCriticNet : AutodiffModule + Clone + ModuleDisplay {
    /// observation
    type Obs: Batchable;
    /// parameters which actor produces. Can be used for creating the distributions.
    type Params;
    /// returns only the params
    fn params(&self, obs: Self::Obs) -> Self::Params;
    /// returns only the state values
    fn values(&self, obs: Self::Obs) -> Tensor<1>;
    /// returns the params (logits, (mean, std) or whatever) and a single value. The user must overload this function when implementing an encoder-shared actor critic
    fn forward(&self, obs: Self::Obs) -> (Self::Params, Tensor<1>) {
        (self.params(obs.clone()), self.values(obs))
    }

    /// returns whether current network is for shared encoder or separated encoder
    fn encoder_type(&self) -> EncoderType;
}