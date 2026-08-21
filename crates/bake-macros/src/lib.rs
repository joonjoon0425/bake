use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{Data, DeriveInput, Field, Fields, Ident, parse_macro_input};

#[derive(Default)]
struct FieldOpts {
    skip: bool,
    size: bool,
    device: bool,
}

/// `#[batchable(skip)]` / `#[batchable(size)]`
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
                o.size = true;
                Ok(())
            } else if meta.path.is_ident("device") {
                o.device = true;
                Ok(())
            }
            else {
                Err(meta.error("unknown option; expected `skip` or `size`"))
            }
        })?;
    }
    if o.skip && o.size {
        return Err(syn::Error::new_spanned(
            f,
            "`skip` and `size` are mutually exclusive",
        ));
    }
    Ok(o)
}

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

    // Three kinds of field:
    //   batched  - delegated to Batchable
    //   skipped  - carried through untouched (first item wins on stack)
    //   size     - a plain usize holding the batch length; never delegated
    let mut batched = Vec::new();
    let mut skipped = Vec::new();
    let mut size_field: Option<Ident> = None;
    let mut device_field: Option<Ident> = None;

    for f in fields {
        let ident = f.ident.clone().unwrap();
        let o = match field_opts(f) {
            Ok(o) => o,
            Err(e) => return e.to_compile_error().into(),
        };

        if o.size {
            if size_field.is_some() {
                return syn::Error::new_spanned(f, "only one field may be marked `size`")
                    .to_compile_error()
                    .into();
            }
            size_field = Some(ident.clone()); // NOT pushed into batched
        } else if o.skip {
            skipped.push(ident.clone());
        } else {
            batched.push(ident.clone());
        }

        if o.device {
            if device_field.is_some() {
                return syn::Error::new_spanned(f, "only one field may be marked `device`")
                    .to_compile_error()
                    .into();
            }
            device_field = Some(ident.clone());
        }
    }

    if batched.is_empty() {
        return syn::Error::new_spanned(
            name,
            "at least one field must be batched (all fields are `skip`/`size`)",
        )
        .to_compile_error()
        .into();
    }

    let bufs: Vec<_> = batched.iter().map(|n| format_ident!("__b_{}", n)).collect();
    let holds: Vec<_> = skipped.iter().map(|n| format_ident!("__s_{}", n)).collect();

    // With an explicit `size` field the length is stored; otherwise ask the
    // first batched field. `size_init` fills that field in stack/select.
    let (batch_size_body, size_init_stack, size_init_select) = match &size_field {
        Some(f) => (
            quote! {
                let Self { #f, .. } = self;
                *#f
            },
            quote! { #f: n, },
            quote! { #f: Batchable::batch_size(&idx), },
        ),
        None => {
            let first = &batched[0];
            (
                quote! { Batchable::batch_size(&self.#first) },
                quote! {},
                quote! {},
            )
        }
    };

    let device_field = device_field.unwrap_or_else(|| batched[0].clone());

    quote! {
        impl #impl_generics Batchable for #name #ty_generics #where_clause {
            fn concat(items: ::std::vec::Vec<Self>) -> Self {
                assert!(!items.is_empty(), "Batchable::concat on an empty Vec");
                let n = items.len();

                #( let mut #bufs = ::std::vec::Vec::with_capacity(n); )*
                #( let mut #holds = ::core::option::Option::None; )*

                for it in items {
                    // `..` swallows the size field, if any.
                    let Self { #(#batched,)* #(#skipped,)* .. } = it;
                    #( #bufs.push(#batched); )*
                    #( if #holds.is_none() {
                           #holds = ::core::option::Option::Some(#skipped);
                       } )*
                }

                Self {
                    #( #batched: Batchable::concat(#bufs), )*
                    #( #skipped: #holds.unwrap(), )*
                    #size_init_stack
                }
            }

            fn batch_size(&self) -> usize {
                #batch_size_body
            }

            fn select(self, idx: Tensor<1, Int>) -> Self {
                let Self { #(#batched,)* #(#skipped,)* .. } = self;
                Self {
                    #( #batched: Batchable::select(#batched, idx.clone()), )*
                    #( #skipped, )*
                    #size_init_select
                }
            }

            fn device(&self) -> burn::tensor::Device {
                Batchable::device(&self.#device_field)
            }
        }
    }
    .into()
}