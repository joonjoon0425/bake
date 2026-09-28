//! The attribute macro for qnet
//! 

use proc_macro2::TokenStream;
use quote::quote;
use syn::{ItemImpl, Result, meta::ParseNestedMeta};
use crate::{discrete_action_value::{dueling, plain}, utils::*};

/// the qnetwork options
#[derive(Default)]
pub(crate) struct QNetOptions {
    dueling: bool,
    // distributional: Option<>,
}

impl QNetOptions {
    pub fn parse(&mut self, meta: ParseNestedMeta) -> syn::parse::Result<()> {
        if meta.path.is_ident("dueling") {
            self.dueling = true;
        }
        // else if meta.path.is_ident("distributional") {
        //     self.distributional = true;
        // }
        else {
            return syn::Result::Err(meta.error("Unsupported qnet property. The supported properties are: dueling"))
        }
        Ok(())
    }
}

pub(crate) fn expand(opts: &QNetOptions, item: &ItemImpl) -> Result<TokenStream> {
    // // allowed for only the inherent impl block
    // if item.trait_.is_some() {
    //     return spanned_error(item.span(), "The qnet attribute macro cannot be applied to non-inherent impl block");
    // }
    let self_ty = &item.self_ty;
    let generics = &item.generics;
    let f = find_func("forward", item)?;
    let obs_ty = obs_type(&f.sig)?;
    let out_span = output_span(&f.sig);

    // the `Net` implementation
    let net_impl = net_impl(generics, self_ty, obs_ty);                         

    let action_value_impl = if opts.dueling {
        dueling(&f.sig.ident, generics, self_ty, out_span)
    } else {
        plain(&f.sig.ident, generics, self_ty, out_span)
    };

    let mut item_original = quote! { #item };
    item_original.extend(net_impl);
    item_original.extend(action_value_impl);
    return Ok(item_original);                                
}
