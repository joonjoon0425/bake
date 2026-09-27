//! Utility functions for attribute macro implementations
//! 
use proc_macro2::Span;
use quote::quote;
use syn::{FnArg, Generics, ImplItem, ImplItemFn, ItemImpl, ReceiverKind, Result, Signature, Type, spanned::Spanned};

/// find a function named 'forward'
pub(crate) fn find_forward(item: &ItemImpl) -> Result<&ImplItemFn> {
    for impl_item in item.items.iter() {
        match impl_item {
            ImplItem::Fn(f) => {
                if f.sig.ident == "forward" {
                    return Ok(f);
                }
            },
            _ => {}
        }
    }
    spanned_error(item.span(), "The attribute macro requires function `forward` to be implemented")
}

/// always creates an error with given msg and span
pub(crate) fn spanned_error<T>(span: Span, msg: &'static str) -> Result<T> {
    syn::Result::Err(syn::Error::new(span, msg))
}

/// check if the receiver has exactly the type &self
pub(crate) fn check_receiver(arg: &FnArg) -> Result<()> {
    match arg {
        syn::FnArg::Receiver(re) => {
            match re.kind {
                ReceiverKind::Reference(_, _, None) => { Ok(()) },
                _ => {
                    return syn::Result::Err(syn::Error::new(arg.span(), "The function `forward` must have following signature: pub fn forward(&self, obs: <observation type>, ...) -> ..."))
                }
            }
        },
        _ => {
            return syn::Result::Err(syn::Error::new(arg.span(), "The function `forward` must have following signature: pub fn forward(&self, obs: <observation type>, ...) -> ..."))
        }
    }
}

/// return the observation type
pub(crate) fn obs_type(sig: &Signature) -> Result<&Type> {
    for (i, arg) in sig.inputs.iter().enumerate() {
        if i == 0 { check_receiver(arg)? }
        else if i == 1 {
            match arg {
                FnArg::Typed(ty) => return Ok(&ty.ty),
                _ => { return spanned_error(arg.span(), "The function `forward` must have following signature: pub fn forward(&self, obs: <observation type>, ...) -> ...") }
            }
        }
    }
    // the forward function do not have obs as its second input
    spanned_error(sig.span(), "The function `forward` must have following signature: pub fn forward(&self, obs: <observation type>, ...) -> ...")
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
        impl #impl_generics ::bake_deep::contract::basic::Network for #self_ty #where_clause {
            type Obs = #obs_ty;
        }
    }
}
