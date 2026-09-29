//! This code was written by Claude, since I don't have enough knowledge about PROC MACRO. Only the batchable.rs and lib.rs is written by claude in this proc-macro lib.
//! `#[derive(Batchable)]`의 구현.
//!
//! 진입점(`derive_batchable`)은 크레이트 루트(`lib.rs`)에 있어야 하므로,
//! 이 모듈은 파싱이 끝난 `DeriveInput`을 받아 생성 코드를 돌려주는 일만 한다.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Data, DeriveInput, Field, Fields, Ident};

#[derive(Default)]
struct FieldOpts {
    skip: bool,
}

/// `#[batchable(skip)]`
///
/// `size`/`device`는 제거되었다. 둘 다 `()`와 `Unconstrained`가 길이·디바이스를
/// 답하지 못해 생긴 우회로였는데, `len()`이 `Option<usize>`가 되고 `device()`가
/// 트레잇에서 빠지면서 필요가 없어졌다.
fn field_opts(f: &Field) -> syn::Result<FieldOpts> {
    let mut o = FieldOpts::default();
    for attr in &f.attrs {
        if !attr.path().is_ident("batchable") {
            continue;
        }
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("skip") {
                o.skip = true;
                Ok(())
            } else {
                Err(meta.error("unknown option; expected `skip`"))
            }
        })?;
    }
    Ok(o)
}

/// 파싱된 struct에 대한 `impl Batchable`을 생성한다.
///
/// 에러는 `?`로 올려보내고, 컴파일 에러로 바꾸는 일은 진입점에서 한 번만 한다.
pub(crate) fn expand(ast: &DeriveInput) -> syn::Result<TokenStream> {
    let name = &ast.ident;
    let (impl_generics, ty_generics, where_clause) = ast.generics.split_for_impl();

    let fields = match &ast.data {
        Data::Struct(s) => match &s.fields {
            Fields::Named(f) => &f.named,
            _ => {
                return Err(syn::Error::new_spanned(
                    name,
                    "Batchable requires a struct with named fields",
                ));
            }
        },
        _ => {
            return Err(syn::Error::new_spanned(
                name,
                "Batchable can only be derived for structs",
            ));
        }
    };

    let mut batched: Vec<Ident> = Vec::new();
    let mut skipped: Vec<Ident> = Vec::new();

    for f in fields {
        let ident = f.ident.clone().unwrap();
        let o = field_opts(f)?;

        if o.skip {
            skipped.push(ident);
        } else {
            batched.push(ident);
        }
    }

    if batched.is_empty() {
        return Err(syn::Error::new_spanned(
            name,
            "at least one field must be batched (all fields are `skip`)",
        ));
    }

    let bufs: Vec<_> = batched.iter().map(|n| format_ident!("__b_{}", n)).collect();
    let holds: Vec<_> = skipped.iter().map(|n| format_ident!("__s_{}", n)).collect();

    Ok(quote! {
        impl #impl_generics Batchable for #name #ty_generics #where_clause {
            fn batch_size(&self) -> ::core::option::Option<usize> {
                ::core::option::Option::None
                #( .or_else(|| Batchable::batch_size(&self.#batched)) )*
            }

            fn cat(items: ::std::vec::Vec<Self>) -> Self {
                assert!(!items.is_empty(), "Batchable::cat on an empty Vec");
                let n = items.len();

                #( let mut #bufs = ::std::vec::Vec::with_capacity(n); )*
                #( let mut #holds = ::core::option::Option::None; )*

                for it in items {
                    let Self { #(#batched,)* #(#skipped,)* } = it;
                    #( #bufs.push(#batched); )*
                    #( if #holds.is_none() {
                           #holds = ::core::option::Option::Some(#skipped);
                       } )*
                }

                Self {
                    #( #batched: Batchable::cat(#bufs), )*
                    #( #skipped: #holds.unwrap(), )*
                }
            }

            fn select(self, idx: Tensor<1, Int>) -> Self {
                let Self { #(#batched,)* #(#skipped,)* } = self;
                Self {
                    #( #batched: Batchable::select(#batched, idx.clone()), )*
                    #( #skipped, )*
                }
            }

            fn slice(self, range: ::core::ops::Range<usize>) -> Self {
                let Self { #(#batched,)* #(#skipped,)* } = self;
                Self {
                    #( #batched: Batchable::slice(#batched, range.clone()), )*
                    #( #skipped, )*
                }
            }

            fn detach(self) -> Self {
                let Self { #(#batched,)* #(#skipped,)* } = self;
                Self {
                    #( #batched: Batchable::detach(#batched), )*
                    #( #skipped, )*
                }
            }

            fn assign_inplace(&mut self, data: Self, index: usize) {
                #(
                    Batchable::assign_inplace(&mut self.#batched, data.#batched, index);
                )*
            }

            fn zeros_like(capacity: usize, data: &Self, device: &Device) -> Self {
                Self {
                    #(
                        #batched: Batchable::zeros_like(capacity, &data.#batched, device),
                    )*
                    #(
                        #skipped: data.#skipped.clone(),
                    )*
                }
            }

            fn to_device(self, device: &Device) -> Self {
                Self {
                    #(
                        #batched: Batchable::to_device(self.#batched, device),
                    )*
                    #(
                        #skipped: self.#skipped
                    )*
                }
            }

            fn into_autodiff(self) -> Self {
                Self {
                    #(
                        #batched: Batchable::into_autodiff(self.#batched),
                    )*
                    #(
                        #skipped: self.#skipped
                    )*
                }
            }
        }
    })
}