//! The attribute macro for `Policy`
//! 

use proc_macro2::{Span, TokenStream};
use quote::{quote, quote_spanned};
use syn::{Generics, Ident, ItemImpl, Result, Type, meta::ParseNestedMeta, spanned::Spanned};

use crate::utils::{find_func, net_impl, obs_type, output_span, spanned_error, update_policy_impl};

/// policy options
#[derive(Default)]
pub(crate) struct PolicyOptions {
    distribution: Option<Type>
}

impl PolicyOptions {
    pub fn parse(&mut self, meta: ParseNestedMeta) -> syn::parse::Result<()> {
        if meta.path.is_ident("distribution") {
            if self.distribution.is_some() {
                return syn::Result::Err(meta.error("The distribution option cannot be applied twice"))
            } else {
                self.distribution = Some(meta.value()?.parse()?)
            }
        }
        // else if meta.path.is_ident("distributional") {
        //     self.distributional = true;
        // }
        else {
            return syn::Result::Err(meta.error("Unsupported field attribute. The supported field attributes are: distribution"))
        }
        Ok(())
    }
}


pub(crate) fn expand(opts: &PolicyOptions, item: &ItemImpl) -> Result<TokenStream> {
    // check if the distribution is not None
    if opts.distribution.is_none() {
        return spanned_error(item.span(), "The policy attribute macro requires a distribution to be specified")
    }

    let self_ty = &item.self_ty;
    let generics = &item.generics;
    let f = find_func("forward", item)?;
    let obs_ty = obs_type(&f.sig)?;
    let out_span = output_span(&f.sig);
    let dist_ty = opts.distribution.as_ref().unwrap();
    // the `Network` implementation
    let net_impl = net_impl(generics, self_ty, obs_ty);                         

    let policy_impl = policy_impl(&f.sig.ident, generics, self_ty, dist_ty, out_span);

    let update_impl = update_policy_impl(generics, self_ty);

    let expanded = quote! {
        #item
        #net_impl
        #policy_impl
        #update_impl
    };
    return Ok(expanded)         
}

pub(crate) fn policy_impl(func_name: &Ident, generics: &Generics, self_ty: &Type, dist_ty: &Type, out_span: Span) -> TokenStream {
    let (impl_generics, _, where_clause) = generics.split_for_impl();

    let make_dist_out = quote_spanned! {
        out_span =>
        C::create_distribution(params, constraint)
    };

    quote! {
        impl #impl_generics ::bake_deep::contract::basic::Policy for #self_ty #where_clause {
            type Dist = #dist_ty;

            fn dist<C: ::bake_deep::distribution::PossibleConstraint<Self::Dist>>(&self, obs: Self::Obs, constraint: C) -> Self::Dist {
                let params = Self::#func_name(self, obs);
                #make_dist_out
            }
        }

    }
}