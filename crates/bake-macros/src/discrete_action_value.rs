//! impl of `DiscreteActionValue` trait
//! 

use proc_macro2::{Span, TokenStream};
use quote::{quote, quote_spanned};
use syn::{Generics, Type};

pub(crate) fn dueling(generics: &Generics, self_ty: &Type, out_span: Span) -> TokenStream {
    let (impl_generics, _, where_clause) = generics.split_for_impl();
    // dueling output check
    let dueling_out = quote_spanned! {
        out_span =>
        let __out: (::burn::prelude::Tensor<1>, ::burn::prelude::Tensor<2>) = Self::forward(self, obs);
    };
    quote! {
        impl #impl_generics ::bake_deep::contract::basic::DiscreteActionValue for #self_ty #where_clause {
            fn action_values<C: ::bake_deep::constraint::discrete_constraint::DiscreteConstraint>(&self, obs: Self::Obs, constraint: C) -> ::burn::prelude::Tensor<2> {
                #dueling_out
                let (value, advantage) = __out;
                let mean = constraint.clone().mean_dim(1, advantage.clone());
                constraint.apply(value.unsqueeze_dim(1) + advantage - mean, -1e9)
            }
        }
    }
}

pub(crate) fn plain(generics: &Generics, self_ty: &Type, out_span: Span) -> TokenStream {
    let (impl_generics, _, where_clause) = generics.split_for_impl();
    // dueling output check
    let plain_out = quote_spanned! {
        out_span =>
        let __out: ::burn::prelude::Tensor<2> = Self::forward(self, obs);
    };
    quote! {
        impl #impl_generics ::bake_deep::contract::basic::DiscreteActionValue for #self_ty #where_clause {
            fn action_values<C: ::bake_deep::constraint::discrete_constraint::DiscreteConstraint>(&self, obs: Self::Obs, constraint: C) -> ::burn::prelude::Tensor<2> {
                #plain_out
                let action_vales = __out;
                constraint.apply(action_vales, -1e9)
            }
        }
    }
}