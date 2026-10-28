use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Data, DeriveInput, Fields, Ident};

pub(crate) fn expand(ast: &DeriveInput) -> syn::Result<TokenStream> {
    let name = &ast.ident;
    let (impl_generics, ty_generics, where_clause) = ast.generics.split_for_impl();

    let fields = match &ast.data {
        Data::Struct(s) => match &s.fields {
            Fields::Named(f) => &f.named,
            _ => {
                return Err(syn::Error::new_spanned(name, "WithBatchDim requires a struct with named fields"));
            }
        },
        _ => {
            return Err(syn::Error::new_spanned(
                name,
                "WithBatchDim can only be derived for struct currently"
            ));
        }
    };

    let field_idents: Vec<Ident> = fields.iter().map(|f| f.ident.clone().unwrap()).collect();
    let first = field_idents[0].clone();
    let bufs: Vec<Ident> = field_idents.iter().map(|i| format_ident!("__f_{}", i)).collect();
    let first_buf = bufs[0].clone();
    
    Ok(quote! {
        impl #impl_generics ::bake_rl::core::data::batch_dim::WithBatchDim for #name #ty_generics #where_clause {
            fn batch_shape(&self) -> ::bake_rl::core::data::batch_dim::BatchShape {
                self.#first.batch_shape()
            }

            fn reshape(self, shape: ::bake_rl::core::data::batch_dim::BatchShape) -> Self {
                Self {
                    #( #field_idents: ::bake_rl::core::data::batch_dim::WithBatchDim::reshape(self.#field_idents, shape.clone()), )*
                }
            }

            fn squeeze(self, dim: isize) -> Self {
                Self {
                    #( #field_idents: ::bake_rl::core::data::batch_dim::WithBatchDim::squeeze(self.#field_idents, dim), )*
                }
            }

            fn unsqueeze(self, dim: isize) -> Self {
                Self {
                    #( #field_idents: ::bake_rl::core::data::batch_dim::WithBatchDim::unsqueeze(self.#field_idents, dim), )*
                }
            }

            fn select(self, dim: isize, idx: ::bake_rl::burn::prelude::Tensor<1, ::bake_rl::burn::prelude::Int>) -> Self {
                Self {
                    #(#field_idents: ::bake_rl::core::data::batch_dim::WithBatchDim::select(self.#field_idents, dim, ::core::clone::Clone::clone(&idx)), )*
                }
            }

            fn assign_inplace(&mut self, data: Self, idx: usize) {
                #( ::bake_rl::core::data::batch_dim::WithBatchDim::assign_inplace(&mut self.#field_idents, data.#field_idents, idx); )*
            }

            fn stack(dat: Vec<Self>) -> Self {
                let n = dat.len();
                #(let mut #bufs = ::std::vec::Vec::with_capacity(n); )*

                for d in dat {
                    let Self { #(#field_idents, )* } = d;
                    #(#bufs.push(#field_idents);)*
                }

                Self {
                    #(#field_idents: ::bake_rl::core::data::batch_dim::WithBatchDim::stack(#bufs), )*
                }
            }

            fn unstack(self) -> Vec<Self> {
                #(let mut #bufs = ::bake_rl::core::data::batch_dim::WithBatchDim::unstack(self.#field_idents); )*
                let __n = #first_buf.len();
                #(assert_eq!(#bufs.len(), __n, "unstack: field length mismatch"); )*

                #(let mut #bufs = #bufs.into_iter(); )*
                (0..__n).map(|_| Self { #(#field_idents: #bufs.next().unwrap(), )* }).collect()
            }
            
            fn cat(dat: Vec<Self>, dim: isize) -> Self {
                let n = dat.len();
                #(let mut #bufs = ::std::vec::Vec::with_capacity(n); )*

                for d in dat {
                    let Self { #(#field_idents, )* } = d;
                    #(#bufs.push(#field_idents);)*
                }

                Self {
                    #(#field_idents: ::bake_rl::core::data::batch_dim::WithBatchDim::cat(#bufs, dim), )*
                }
            }

            fn slice(self, dim: isize, range: std::ops::Range<usize>) -> Self {
                Self {
                    #(#field_idents: ::bake_rl::core::data::batch_dim::WithBatchDim::slice(self.#field_idents, dim, ::core::clone::Clone::clone(&range)), )*
                }
            }

            fn zeros_like(capacity: usize, data: &Self) -> Self {
                Self {
                    #(#field_idents: ::bake_rl::core::data::batch_dim::WithBatchDim::zeros_like(capacity, &data.#field_idents), )*
                }
            }
        }
    })
}