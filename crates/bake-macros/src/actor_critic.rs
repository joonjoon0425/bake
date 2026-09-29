//! attribute macro implementation for actor critic
//! 

use proc_macro2::{Span, TokenStream};
use quote::{quote, quote_spanned};
use syn::{Generics, Ident, ItemImpl, Result, Type, meta::ParseNestedMeta, spanned::Spanned};

use crate::{policy, state_value, utils::{find_func, net_impl, obs_type, output_span, spanned_error, update_impl, update_separated_ac_impl}};
#[derive(Default)]
pub(crate) struct ActorCriticOptions {
    distribution: Option<Type>,
    encoder_sharing: bool,
}

impl ActorCriticOptions {
    pub fn parse(&mut self, meta: ParseNestedMeta) -> syn::parse::Result<()> {
        if meta.path.is_ident("distribution") {
            if self.distribution.is_some() {
                return syn::Result::Err(meta.error("The distribution option cannot be applied twice"))
            } else {
                self.distribution = Some(meta.value()?.parse()?)
            }
        } else if meta.path.is_ident("encoder_shared") {
            self.encoder_sharing = true;
        } else {
            return syn::Result::Err(meta.error("Unsupported field attribute. The supported field attributes are: distribution, encoder_shared"))
        }
        Ok(())
    }
}

pub(crate) fn expand(opts: &ActorCriticOptions, item: &ItemImpl) -> Result<TokenStream> {
    // check if the distribution is not None
    if opts.distribution.is_none() {
        return spanned_error(item.span(), "The actor_critic attribute macro requires a distribution to be specified")
    }

    let self_ty = &item.self_ty;
    let generics = &item.generics;
    let actor_f = find_func("actor", item)?;
    let critic_f = find_func("critic", item)?;
    let actor_out_span = output_span(&actor_f.sig);
    let critic_out_span = output_span(&critic_f.sig);
    let dist_ty = opts.distribution.as_ref().unwrap();
    let obs_ty = obs_type(&actor_f.sig)?;

    // the `Network` implementation
    let net_impl = net_impl(generics, &self_ty, obs_ty);

    let actor_critic_impl = if opts.encoder_sharing {
        let actor_critic_f = find_func("actor_critic", item)?;
        let actor_critic_out_span = output_span(&actor_critic_f.sig);
        let imp = encoder_shared(&actor_critic_f.sig.ident, &actor_f.sig.ident, &critic_f.sig.ident, generics, self_ty, dist_ty, actor_critic_out_span, actor_out_span, critic_out_span);
        let update_impl = update_impl(generics, self_ty);
        quote! {
            #imp
            #update_impl
        }
    } else {
        let imp = encoder_separated(&actor_f.sig.ident, &critic_f.sig.ident, generics, self_ty, dist_ty, actor_out_span, critic_out_span);
        let update_impl = update_separated_ac_impl(generics, self_ty);
        quote! {
            #imp
            #update_impl
        }
    };

    let expanded = quote! {
        #item
        #net_impl
        #actor_critic_impl
    };
    return Ok(expanded);      
}

pub(crate) fn encoder_separated(actor_f: &Ident, critic_f: &Ident, generics: &Generics, self_ty: &Type, dist_ty: &Type, actor_out_span: Span, critic_out_span: Span) -> TokenStream {
    let (impl_generics, _, where_clause) = generics.split_for_impl();
    let actor = policy::policy_impl(actor_f, generics, self_ty, dist_ty, actor_out_span);
    let critic = state_value::state_value(critic_f, generics, self_ty, critic_out_span);
    quote! {
        #actor
        #critic
        impl #impl_generics ::bake_deep::contract::compound::ActorCritic for #self_ty #where_clause {
            fn dist_and_state_value<C: ::bake_deep::distribution::PossibleConstraint<Self::Dist>>(&self, obs: Self::Obs, constraint: C) -> (Self::Dist, ::bake_deep::burn::prelude::Tensor<1>) {
                let dist = ::bake_deep::contract::basic::Policy::dist(self, obs.clone(), constraint);
                let state_value = ::bake_deep::contract::basic::StateValue::state_value(self, obs);
                (dist, state_value)
            }
        }
    }
}

pub(crate) fn encoder_shared(actor_critic_f: &Ident, actor_f: &Ident, critic_f: &Ident, generics: &Generics, self_ty: &Type, dist_ty: &Type, actor_critic_out_span: Span, actor_out_span: Span, critic_out_span: Span) -> TokenStream {
    let (impl_generics, _, where_clause) = generics.split_for_impl();
    let actor = policy::policy_impl(actor_f, generics, self_ty, dist_ty, actor_out_span);
    let critic = state_value::state_value(critic_f, generics, self_ty, critic_out_span);

    let actor_critic_out = quote_spanned! {
        actor_critic_out_span =>
        let (__params, __state_value) = Self::#actor_critic_f(self, obs);
        let __dist = C::create_distribution(__params, constraint);
    };
    quote! {
        #actor
        #critic
        impl #impl_generics ::bake_deep::contract::compound::ActorCritic for #self_ty #where_clause {
            fn dist_and_state_value<C: ::bake_deep::distribution::PossibleConstraint<Self::Dist>>(&self, obs: Self::Obs, constraint: C) -> (Self::Dist, ::bake_deep::burn::prelude::Tensor<1>) {
                #actor_critic_out
                (__dist, __state_value)
            }
        }
    }
}