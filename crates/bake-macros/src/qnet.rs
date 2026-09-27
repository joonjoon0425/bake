//! The attribute macro for discrete qnet
//! 

use proc_macro2::TokenStream;
use quote::{quote, quote_spanned};
use syn::{ItemImpl, Result, meta::ParseNestedMeta, spanned::Spanned};
use crate::utils::*;

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
    // allowed for only the inherent impl block
    if item.trait_.is_some() {
        return spanned_error(item.span(), "The qnet attribute macro cannot be applied to non-inherent impl block");
    }
    let self_ty = &item.self_ty;
    let generics = &item.generics;
    let (impl_generics, _, where_clause) = generics.split_for_impl();
    let f = find_forward(item)?;
    let obs_ty = obs_type(&f.sig)?;
    let out_span = output_span(&f.sig);

    // the `Net` implementation
    let net_impl = net_impl(generics, self_ty, obs_ty);

    let dueling_out = quote_spanned! {
        out_span =>
        let __out: (::burn::prelude::Tensor<1>, ::burn::prelude::Tensor<2>) = Self::forward(self, obs);
    };                                
                                    
    let plain_out = quote_spanned! {
        out_span =>
        let __out: ::burn::prelude::Tensor<2> = Self::forward(self, obs);
    };                                

    let action_value_impl = if opts.dueling {
        quote! {
            impl #impl_generics ::bake_deep::experimental::contract::basic::DiscreteActionValue for #self_ty #where_clause {
                fn action_values<C: ::bake_deep::constraint::discrete_constraint::DiscreteConstraint>(&self, obs: Self::Obs, constraint: C) -> ::burn::prelude::Tensor<2> {
                    #dueling_out
                    let (value, advantage) = __out;
                    let mean = constraint.clone().mean_dim(1, advantage.clone());
                    constraint.apply(value.unsqueeze_dim(1) + advantage - mean, -1e9)
                }
            }
        }
    } else {
        quote! {
            impl #impl_generics ::bake_deep::experimental::contract::basic::DiscreteActionValue for #self_ty #where_clause {
                fn action_values<C: ::bake_deep::constraint::discrete_constraint::DiscreteConstraint>(&self, obs: Self::Obs, constraint: C) -> ::burn::prelude::Tensor<2> {
                    #plain_out
                    let action_vales = __out;
                    constraint.apply(action_vales, -1e9)
                }
            }
        }
    };
    let mut item_original = quote! { #item };
    item_original.extend(net_impl);
    item_original.extend(action_value_impl);
    return Ok(item_original);                                
}
