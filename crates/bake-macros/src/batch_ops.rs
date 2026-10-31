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
                return Err(syn::Error::new_spanned(name, "BatchOps requires a struct with named fields"));
            }
        },
        _ => {
            return Err(syn::Error::new_spanned(
                name,
                "BatchOps can only be derived for struct currently"
            ));
        }
    };

    let field_idents: Vec<Ident> = fields.iter().map(|f| f.ident.clone().unwrap()).collect();
    let bufs: Vec<Ident> = field_idents.iter().map(|i| format_ident!("__f_{}", i)).collect();
    let first_buf = bufs[0].clone();
    
    Ok(quote! {
        impl #impl_generics ::bake_rl::core::data::batch_ops::BatchOps for #name #ty_generics #where_clause {
            fn check_batch(&self, bs: &[usize]) -> bool {
                let mut b = true;
                #(b = b && ::bake_rl::core::data::batch_ops::BatchOps::check_batch(self.#field_idents, bs);)*
            }

            fn reshape_batch(self, bs: &[usize], new_bs: &[usize]) -> Self {
                Self {
                    #( #field_idents: ::bake_rl::core::data::batch_ops::BatchOps::reshape(self.#field_idents, bs, new_bs), )*
                }
            }

            fn select_batch(self, bs: &[usize], dim: isize, idx: ::bake_rl::burn::prelude::Tensor<1, ::bake_rl::burn::prelude::Int>) -> Self {
                Self {
                    #(#field_idents: ::bake_rl::core::data::batch_ops::BatchOps::select(self.#field_idents, bs, dim, ::core::clone::Clone::clone(&idx)), )*
                }
            }

            fn assign_inplace_batch(&mut self, bs: &[usize], dim: isize, idx: usize, src: Self, src_bs: &[usize]) {
                #( ::bake_rl::core::data::batch_ops::BatchOps::assign_inplace_batch(&mut self.#field_idents, bs, dim, idx, src.#field_idents, src_bs); )*
            }

            fn unbind_batch(self, bs: &[usize], dim: isize) -> Vec<Self> {
                #(let mut #bufs = ::bake_rl::core::data::batch_ops::BatchOps::unbind_batch(self.#field_idents, bs, dim); )*
                let __n = #first_buf.len();
                #(assert_eq!(#bufs.len(), __n, "unstack: field length mismatch"); )*

                #(let mut #bufs = #bufs.into_iter(); )*
                (0..__n).map(|_| Self { #(#field_idents: #bufs.next().unwrap(), )* }).collect()
            }
            
            fn cat_batch(bs: &[Vec<usize>], dim: isize, dat: Vec<Self>) -> Self {
                let n = dat.len();
                #(let mut #bufs = ::std::vec::Vec::with_capacity(n); )*

                for d in dat {
                    let Self { #(#field_idents, )* } = d;
                    #(#bufs.push(#field_idents);)*
                }

                Self {
                    #(#field_idents: ::bake_rl::core::data::batch_ops::BatchOps::cat_batch(bs, dim, #bufs), )*
                }
            }

            fn slice_batch(self, bs: &[usize], dim: isize, range: std::ops::Range<usize>) -> Self {
                Self {
                    #(#field_idents: ::bake_rl::core::data::batch_ops::BatchOps::slice(self.#field_idents, bs, dim, ::core::clone::Clone::clone(&range)), )*
                }
            }

            fn zeros_like_batch(bs: &[usize], dim: isize, capacity: usize, src: &Self) -> Self {
                Self {
                    #(#field_idents: ::bake_rl::core::data::batch_ops::BatchOps::zeros_like(bs, dim, capacity, &src.#field_idents), )*
                }
            }

            fn autodiff(self) -> Self {
                Self {
                    #(#field_idents: ::bake_rl::core::data::batch_ops::BatchOps::autodiff(self), )*
                }
            }
        }
    })
}