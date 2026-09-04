//! This code was written by Claude, since I don't have enough knowledge about PROC MACRO

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{Data, DeriveInput, Field, Fields, Ident, parse_macro_input};

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
            } else if meta.path.is_ident("size") {
                Err(meta.error(
                    "`size` was removed; `len()` now returns `Option<usize>` and \
                     derived types take the first `Some` among batched fields",
                ))
            } else if meta.path.is_ident("device") {
                Err(meta.error(
                    "`device` was removed; `Batchable` no longer has `device()`. \
                     Use the `HasDevice` trait on the field you need it from",
                ))
            } else {
                Err(meta.error("unknown option; expected `skip`"))
            }
        })?;
    }
    Ok(o)
}

/// `Batchable`을 파생한다.
///
/// 필드는 두 종류다.
/// - **batched**: `Batchable`로 위임
/// - **skipped** (`#[batchable(skip)]`): 손대지 않고 통과. `cat`에서는 첫 항목이 이긴다
///
/// `len`은 배치 필드 중 첫 `Some`을 취한다. 모든 배치 필드가 길이를 갖지 않으면
/// (예: 전부 `()`) `None`이 된다.
#[proc_macro_derive(Batchable, attributes(batchable))]
pub fn derive_batchable(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    let name = &ast.ident;
    let (impl_generics, ty_generics, where_clause) = ast.generics.split_for_impl();

    let fields = match &ast.data {
        Data::Struct(s) => match &s.fields {
            Fields::Named(f) => &f.named,
            _ => {
                return syn::Error::new_spanned(
                    name,
                    "Batchable requires a struct with named fields",
                )
                .to_compile_error()
                .into();
            }
        },
        _ => {
            return syn::Error::new_spanned(name, "Batchable can only be derived for structs")
                .to_compile_error()
                .into();
        }
    };

    let mut batched: Vec<Ident> = Vec::new();
    let mut skipped: Vec<Ident> = Vec::new();

    for f in fields {
        let ident = f.ident.clone().unwrap();
        let o = match field_opts(f) {
            Ok(o) => o,
            Err(e) => return e.to_compile_error().into(),
        };

        if o.skip {
            skipped.push(ident);
        } else {
            batched.push(ident);
        }
    }

    if batched.is_empty() {
        return syn::Error::new_spanned(
            name,
            "at least one field must be batched (all fields are `skip`)",
        )
        .to_compile_error()
        .into();
    }

    let bufs: Vec<_> = batched.iter().map(|n| format_ident!("__b_{}", n)).collect();
    let holds: Vec<_> = skipped.iter().map(|n| format_ident!("__s_{}", n)).collect();

    quote! {
        impl #impl_generics Batchable for #name #ty_generics #where_clause {
            fn len(&self) -> ::core::option::Option<usize> {
                ::core::option::Option::None
                #( .or_else(|| Batchable::len(&self.#batched)) )*
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
                        #skipped
                    )*
                }
            }
        }
    }
    .into()
}