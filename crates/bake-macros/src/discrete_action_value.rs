//! impl of `DiscreteActionValue` trait
//! 

use proc_macro2::{Span, TokenStream};
use quote::{quote, quote_spanned};
use syn::{Generics, Ident, Type};

pub(crate) fn dueling(func_name: &Ident, generics: &Generics, self_ty: &Type, out_span: Span) -> TokenStream {
    let (impl_generics, _, where_clause) = generics.split_for_impl();
    // dueling output check
    let dueling_out = quote_spanned! {
        out_span =>
        let __out: (::bake_rl::core::data::BatchTensor<1>, ::bake_rl::core::data::BatchTensor<2>) = Self::#func_name(self, obs);
    };
    quote! {
        impl #impl_generics ::bake_rl::deep::contract::DiscreteActionValue for #self_ty #where_clause {
            fn action_values<C: ::bake_rl::core::constraint::discrete::DiscreteConstraint>(&self, obs: Self::Obs, constraint: C) -> ::bake_rl::core::data::BatchTensor<2> {
                #dueling_out
                let (value, advantage) = __out;
                let mean = constraint.clone().mean_dim(advantage.clone());
                let (value, bs) = value.detach_batch_dim();
                let mean = mean.detach_batch_dim().0;
                let advantage = advantage.detach_batch_dim().0;

                constraint.apply((value.unsqueeze_dim(1) + advantage - mean.unsqueeze_dim(1)).attach_batch_dim(bs), -1e9)
            }
        }
    }
}

pub(crate) fn plain(func_name: &Ident, generics: &Generics, self_ty: &Type, out_span: Span) -> TokenStream {
    let (impl_generics, _, where_clause) = generics.split_for_impl();
    // dueling output check
    let plain_out = quote_spanned! {
        out_span =>
        let __out: ::bake_rl::core::data::BatchTensor<2> = Self::#func_name(self, obs);
    };
    quote! {
        impl #impl_generics ::bake_rl::deep::contract::DiscreteActionValue for #self_ty #where_clause {
            fn action_values<C: ::bake_rl::core::constraint::discrete::DiscreteConstraint>(&self, obs: Self::Obs, constraint: C) -> ::bake_rl::core::data::BatchTensor<2> {
                #plain_out
                let action_vales = __out;
                constraint.apply(action_vales, -1e9)
            }
        }
    }
}