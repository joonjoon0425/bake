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

    let bufs: Vec<Ident> = field_idents.iter().map(|i| format_ident!("__f_{}", i)).collect();

    Ok(quote! {
        impl #impl_generics WithBatchDim for #name #ty_generics #where_clause {
            fn batch_shape(&self) -> &::bake_rl::core::data::batch_dim::BatchShape {
                #(field_idents[0]).batch_shape()
            }

            fn reshape(self, shape: ::bake_rl::core::data::batch_dim::BatchShape) -> Self {
                Self {
                    #( #field_idents: WithBatchDim::reshape(self.#field_idents, shape.clone()), )*
                }
            }

            fn squeeze(self, dim: isize) -> Self {
                Self {
                    #( #field_idents: WithBatchDim::squeeze(self.#field_idents, dim), )*
                }
            }

            fn unsqueeze(self, dim: isize) -> Self {
                Self {
                    #( #field_idents: WithBatchDim::unsqueeze(self.#field_idents, dim), )*
                }
            }

            fn assign_inplace(&mut self, data: Self, idx: usize) {
                #( WithBatchDim::assign_inplace(&mut self.#field_idents, data.#field_idents, idx.clone()))*
            }

            fn stack(dat: Vec<Self>) -> Self {
                let n = dat.len();
                # (let #bufs = ::std::vec::Vec::with_capacity(n); )*

                for d in dat {
                    let Self { #(#field_idents, )* } = d;
                    #(#bufs.push(#field_idents);)*
                }

                Self {
                    #(#field_idents: WithBatchDim::stack(#bufs), )*
                }
            }
            
            fn cat(dat: Vec<Self>, dim: isize) -> Self {
                let n = dat.len();
                # (let #bufs = ::std::vec::Vec::with_capacity(n); )*

                for d in dat {
                    let Self { #(#field_idents, )* } = d;
                    #(#bufs.push(#field_idents);)*
                }

                Self {
                    #(#field_idents: WithBatchDim::cat(#bufs, dim), )*
                }
            }

            fn slice(self, dim: isize, range: std::range::Range<usize>) -> Self {
                Self {
                    #(#field_idents: WithBatchDim::slice(#field_idents, dim, range.clone()), )*
                }
            }

            fn select(self, dim: isize, idx: ::burn::prelude::Tensor<1, ::burn::prelude::Int>) -> Self {
                Self {
                    #(#field_idents: WithBatchDim::select(#field_idents, dim, idx.clone()), )*
                }
            }

            fn zeros_like(capacity: usize, data: &Self) -> Self {
                Self {
                    #(#field_idents: WithBatchDim::zeros_like(capacity, &data.#field_idents), )*
                }
            }
        }
    })
}