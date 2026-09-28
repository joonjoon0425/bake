//! implementation of `StateValue`
//! 

use proc_macro2::{Span, TokenStream};
use quote::{quote, quote_spanned};
use syn::{Generics, Ident, Type};

pub(crate) fn state_value(func_name: &Ident, generics: &Generics, self_ty: &Type, out_span: Span) -> TokenStream {
    let (impl_generics, _, where_clause) = generics.split_for_impl();

    let out = quote_spanned! {
        out_span =>
        let __out = Self::#func_name(self, obs);
    };

    quote! {
        impl #impl_generics ::bake_deep::contract::basic::StateValue for #self_ty #where_clause {
            fn state_value(&self, obs: Self::Obs) -> ::bake_deep::burn::prelude::Tensor<1> {
                #out
                __out
            }
        }
    }
}