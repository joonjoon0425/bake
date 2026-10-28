//! This code was written by Claude, since I don't have enough knowledge about PROC MACRO. Only the batchable.rs and lib.rs is written by claude in this proc-macro lib.

use proc_macro::TokenStream;
use syn::{DeriveInput, ItemImpl, parse_macro_input};

mod batch_dim;
mod utils;
mod discrete_action_value;

mod actor_critic;
mod state_value;
mod policy;
mod qnet;

/// derive `WithBatchDim`
#[proc_macro_derive(WithBatchDim)]
pub fn derive_with_batch_dim(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    batch_dim::expand(&ast)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

/// This macro implements following traits;
/// - `Network`
/// - `DiscreteActionValue`
/// # Field Arguments
/// ## Mandatory
/// - No mandatory field arguments
/// ## Optional
/// - `qnet(dueling)`: for dueling dqn methods
/// # Warning
/// - The user must implement `forward` function that returns the action values of given state
/// - The `forward` function must have one of the following signatures:
///     - `pub fn forward(&self, obs: <observation type>) -> Tensor<2>` for plain qnet
///     - `pub fn forward(&self, obs: <observation type>) -> (Tensor<1>, Tensor<2>)` for dueling qnet
#[proc_macro_attribute]
pub fn qnet(args: TokenStream, input: TokenStream) -> TokenStream {
    let mut opts = qnet::QNetOptions::default();
    let qnet_parser = syn::meta::parser(|meta| opts.parse(meta));
    parse_macro_input!(args with qnet_parser);

    let input = parse_macro_input!(input as ItemImpl);
    qnet::expand(&opts, &input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

/// This macro implements following traits;
/// - `Network`
/// - `Policy`
/// # Field Arguments
/// ## Mandatory
/// - `policy(distribution = <distribution name>)`. Followings are the list of possible distributions
///     - `Categorical`
/// ## Optional
/// - No optional field arguments
/// # Warning
/// - The user must implement `forward` function which returns the parameters of the distribution
/// - The `forward` function must have following signature: `pub fn forward(&self, obs: <observation type>) -> <distribution name>::Params`
#[proc_macro_attribute]
pub fn policy(args: TokenStream, input: TokenStream) -> TokenStream {
    let mut opts = policy::PolicyOptions::default();
    let policy_parser = syn::meta::parser(|meta| opts.parse(meta));
    parse_macro_input!(args with policy_parser);

    let input = parse_macro_input!(input as ItemImpl);
    policy::expand(&opts, &input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}


/// This macro implements following traits;
/// - `Network`
/// - `Policy`
/// - `StateValue`
/// - `ActorCritic`
/// # Field Arguments
/// ## Mandatory
/// - `actor_critic(distribution = <distribution name>)`. Followings are the list of possible distributions
///     - `Categorical`
/// ## Optional
/// - `actor_critic(encoder_shared)`
///     - This argument indicates that the actor and critic shares the encoder
///     - If the argument is not indicated, the actor and critic does not share the encoder
/// # Warning
/// - The user must implement `actor` and `critic` function which returns the parameters of distribution and state value of given state, respectively
/// - If the `encoder_shared` field argument is used, the user must implement the `actor_critic` function which returns the parameters of distribution and state value of given state at once.
///     - The user must make sure that the encoder is shared for actor and critic 
#[proc_macro_attribute]
pub fn actor_critic(args: TokenStream, input: TokenStream) -> TokenStream {
    let mut opts = actor_critic::ActorCriticOptions::default();
    let actor_critic_parser = syn::meta::parser(|meta| opts.parse(meta));
    parse_macro_input!(args with actor_critic_parser);

    let input = parse_macro_input!(input as ItemImpl);
    actor_critic::expand(&opts, &input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}
