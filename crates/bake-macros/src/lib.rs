//! This code was written by Claude, since I don't have enough knowledge about PROC MACRO. Only the batchable.rs and lib.rs is written by claude in this proc-macro lib.

use proc_macro::TokenStream;
use syn::{DeriveInput, ItemImpl, parse_macro_input};

mod batchable;
mod utils;
mod discrete_action_value;
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
/// # Warning
/// - The user must implement `forward` function in one's network struct
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

// #[proc_macro_attribute]
// pub fn policy(args: TokenStream, input: TokenStream) -> TokenStream {
    
// }

// #[proc_macro_attribute]
// pub fn actor_critic(args: TokenStream, input: TokenStream) -> TokenStream {
    
// }
