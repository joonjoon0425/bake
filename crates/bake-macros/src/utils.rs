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
        impl #impl_generics ::bake_deep::contract::basic::Network for #self_ty #where_clause {
            type Obs = #obs_ty;
        }
    }
}
