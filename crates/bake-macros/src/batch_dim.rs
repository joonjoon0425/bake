use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Data, DeriveInput, Fields, Ident, Type, parse_quote};

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
    let field_tys: Vec<Type> = fields.iter().map(|f| f.ty.clone()).collect();
    let first = field_idents[0].clone();
    let bufs: Vec<Ident> = field_idents.iter().map(|i| format_ident!("__f_{}", i)).collect();

    // for BatchSelect
    let mut g = ast.generics.clone();
    g.params.push(parse_quote!(__I: ::bake_rl::core::data::batch_dim::BatchIndex));
    {
        let wc = g.make_where_clause();
        for ty in &field_tys {
            wc.predicates.push(parse_quote!(#ty: ::bake_rl::core::data::batch_dim::BatchSelect<__I>));
        }
    }
    let (impl_generics_select, _, where_clause_select) = g.split_for_impl();
    
    Ok(quote! {
        impl #impl_generics ::bake_rl::core::data::batch_dim::WithBatchDim for #name #ty_generics #where_clause {
            fn batch_shape(&self) -> &::bake_rl::core::data::batch_dim::BatchShape {
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

            fn slice(self, dim: isize, range: std::range::Range<usize>) -> Self {
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

        impl #impl_generics_select ::bake_rl::core::data::batch_dim::BatchSelect<__I> for #name #ty_generics #where_clause_select {
            fn select(self, dim: isize, idx: __I) -> Self {
                Self {
                    #(#field_idents: ::bake_rl::core::data::batch_dim::BatchSelect::<__I>::select(self.#field_idents, dim, ::core::clone::Clone::clone(&idx)), )*
                }
            }
        }
    })
}