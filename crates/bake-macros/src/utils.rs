//! Utility functions for attribute macro implementations
//! 
use proc_macro2::Span;
use quote::quote;
use syn::{FnArg, Generics, ImplItem, ImplItemFn, ItemImpl, ReceiverKind, Result, Signature, Type, spanned::Spanned};

/// find a function of given name
pub(crate) fn find_func<'a>(func_name: &'static str, item: &'a ItemImpl) -> Result<&'a ImplItemFn> {
    for impl_item in item.items.iter() {
        match impl_item {
            ImplItem::Fn(f) => {
                if f.sig.ident == func_name {
                    return Ok(f);
                }
            },
            _ => {}
        }
    }
    spanned_error(item.span(), &format!("The attribute macro requires function `{}` to be implemented", func_name))
}

/// always creates an error with given msg and span
pub(crate) fn spanned_error<T>(span: Span, msg: &str) -> Result<T> {
    syn::Result::Err(syn::Error::new(span, msg))
}

/// check if the receiver has exactly the type &self
pub(crate) fn check_receiver(sig: &Signature, arg: &FnArg) -> Result<()> {
    match arg {
        syn::FnArg::Receiver(re) => {
            match re.kind {
                ReceiverKind::Reference(_, _, None) => { Ok(()) },
                _ => {
                    return syn::Result::Err(syn::Error::new(arg.span(), &format!("The function `{}` must have following signature: pub fn {}(&self, obs: <observation type>, ...) -> ...", sig.ident.to_string(), sig.ident.to_string())))
                }
            }
        },
        _ => {
            return syn::Result::Err(syn::Error::new(arg.span(), &format!("The function `{}` must have following signature: pub fn {}(&self, obs: <observation type>, ...) -> ...", sig.ident.to_string(), sig.ident.to_string())))
        }
    }
}

/// return the observation type
pub(crate) fn obs_type(sig: &Signature) -> Result<&Type> {
    for (i, arg) in sig.inputs.iter().enumerate() {
        if i == 0 { check_receiver(sig, arg)? }
        else if i == 1 {
            match arg {
                FnArg::Typed(ty) => return Ok(&ty.ty),
                _ => { return spanned_error(arg.span(), &format!("The function `{}` must have following signature: pub fn {}(&self, obs: <observation type>, ...) -> ...", sig.ident.to_string(), sig.ident.to_string())) }
            }
        }
    }
    // the forward function do not have obs as its second input
    spanned_error(sig.span(), &format!("The function `{}` must have following signature: pub fn {}(&self, obs: <observation type>, ...) -> ...", sig.ident.to_string(), sig.ident.to_string()))
}

/// get the output span for user-friendly error message
pub(crate) fn output_span(sig: &Signature) -> Span {
    return match &sig.output {
        syn::ReturnType::Type(_, ty) => ty.span(),
        syn::ReturnType::Default => sig.span(),
    };
}

/// `Network` trait implementation TokenStream
pub(crate) fn net_impl(generics: &Generics, self_ty: &Type, obs_ty: &Type) -> proc_macro2::TokenStream {
    let (impl_generics, _, where_clause) = generics.split_for_impl();
    quote! {
        impl #impl_generics ::bake_deep::net::network::Network for #self_ty #where_clause {
            type Obs = #obs_ty;
        }
    }
}

/// ordinary `update` function
pub(crate) fn update_impl(generics: &Generics, self_ty: &Type) -> proc_macro2::TokenStream {
    let (impl_generics, _, where_clause) = generics.split_for_impl();
    quote! {
        impl #impl_generics #self_ty #where_clause {
            /// update the network with given learning rate and optimizer
            /// # Warning
            /// - The given network must be on autodiff device, which the loss function does it.
            /// - The given network is moved to inner device after the function call
            pub fn update<N: ::bake_deep::net::network::Network>(net: N, loss: ::bake_deep::burn::prelude::Tensor<1>, lr: f64, opt: &mut ::bake_deep::burn::optim::ModuleOptimizer) -> N {
                ::bake_deep::net::network::update(net, loss, lr, opt).valid()
            }
        }
    }
}

/// policy `update` function
pub(crate) fn update_policy_impl(generics: &Generics, self_ty: &Type) -> proc_macro2::TokenStream {
    let (impl_generics, _, where_clause) = generics.split_for_impl();
    quote! {
        impl #impl_generics #self_ty #where_clause {
            /// update the network with given learning rate and optimizer
            /// # Warning
            /// - The given network must be on autodiff device, which the loss function does it.
            /// - The given network is moved to inner device after the function call
            pub fn update<N: ::bake_deep::net::network::Network>(net: N, loss: ::bake_deep::burn::prelude::Tensor<1>, entropy: ::bake_deep::burn::prelude::Tensor<1>, c_e: f32, lr: f64, opt: &mut ::bake_deep::burn::optim::ModuleOptimizer) -> N {
                let loss = loss - entropy * c_e;
                ::bake_deep::net::network::update(net, loss, lr, opt).valid()
            }
        }
    }
}

/// actor_critic separated `update` function
pub(crate) fn update_separated_ac_impl(generics: &Generics, self_ty: &Type) -> proc_macro2::TokenStream {
    let (impl_generics, _, where_clause) = generics.split_for_impl();
    quote! {
        impl #impl_generics #self_ty #where_clause {
            /// update the network with given learning rate and optimizer
            /// # Warning
            /// - The given network must be on autodiff device, which the loss function does it.
            /// - The given network is moved to inner device after the function call
            pub fn update<N: ::bake_deep::net::network::Network>(
                mut net: N,
                actor_loss: ::bake_deep::burn::prelude::Tensor<1>,
                critic_loss: ::bake_deep::burn::prelude::Tensor<1>,
                entropy: ::bake_deep::burn::prelude::Tensor<1>,
                c_e: f32,
                lr_a: f64,
                lr_c: f64,
                opt_a: &mut ::bake_deep::burn::optim::ModuleOptimizer,
                opt_c: &mut ::bake_deep::burn::optim::ModuleOptimizer,
            ) -> N {
                let loss = actor_loss - entropy * c_e;
                net = ::bake_deep::net::network::update(net, loss, lr_a, opt_a);
                net = ::bake_deep::net::network::update(net, critic_loss, lr_c, opt_c);
                net.valid()
            }
        }
    }
}

/// actor_critic shared `update` function
pub(crate) fn update_shared_ac_impl(generics: &Generics, self_ty: &Type) -> proc_macro2::TokenStream {
    let (impl_generics, _, where_clause) = generics.split_for_impl();
    quote! {
        impl #impl_generics #self_ty #where_clause {
            /// update the network with given learning rate and optimizer
            /// # Warning
            /// - The given network must be on autodiff device, which the loss function does it.
            /// - The given network is moved to inner device after the function call
            pub fn update<N: ::bake_deep::net::network::Network>(
                net: N,
                actor_loss: ::bake_deep::burn::prelude::Tensor<1>,
                critic_loss: ::bake_deep::burn::prelude::Tensor<1>,
                entropy: ::bake_deep::burn::prelude::Tensor<1>,
                c_e: f32,
                c_c: f32,
                lr: f64,
                opt: &mut ::bake_deep::burn::optim::ModuleOptimizer
            ) -> N {
                let loss = actor_loss - entropy * c_e + critic_loss * c_c;
                ::bake_deep::net::network::update(net, loss, lr, opt).valid()
            }
        }
    }
}