//! This code was written by Claude, since I don't have enough knowledge about PROC MACRO. Only the batchable.rs and lib.rs is written by claude in this proc-macro lib.

use proc_macro::TokenStream;
use syn::{DeriveInput, ItemImpl, parse_macro_input};

mod batchable;
mod utils;
mod discrete_action_value;

mod actor_critic;
mod state_value;
mod policy;
mod qnet;
/// `Batchable`을 파생한다.
///
/// 필드는 두 종류다.
/// - **batched**: `Batchable`로 위임
/// - **skipped** (`#[batchable(skip)]`): 손대지 않고 통과. `cat`에서는 첫 항목이 이긴다
///
/// `len`은 배치 필드 중 첫 `Some`을 취한다. 모든 배치 필드가 길이를 갖지 않으면
/// (예: 전부 `()`) `None`이 된다.
#[proc_macro_derive(Batchable, attributes(batchable))]
pub fn derive_batchable(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    batchable::expand(&ast)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

/// This macro implements following traits;
/// - `Network`
/// - `ActionValue`
/// # Field Attributes
/// ## Mandatory
/// - No mandatory field attributes
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
/// # Field Attributes
/// ## Mandatory
/// - `policy(distribution = <distribution name>)`. Followings are the list of possible distributions
///     - `Categorical`
/// ## Optional
/// - No optional field attributes
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
/// # Field Attributes
/// ## Mandatory
/// - `actor_critic(distribution = <distribution name>)`. Followings are the list of possible distributions
///     - `Categorical`
/// ## Optional
/// - `actor_critic(encoder_shared)`
///     - This attribute indicates that the actor and critic shares the encoder
///     - If the attribute is not indicated, the actor and critic does not share the encoder
/// # Warning
/// - The user must implement `actor` and `critic` function which returns the parameters of distribution and state value of given state, respectively
/// - If the `encoder_shared` field attribute is used, the user must implement the `actor_critic` function which returns the parameters of distribution and state value of given state at once.
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
