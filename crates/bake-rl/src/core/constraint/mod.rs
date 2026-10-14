//! Constraint trait for actions
//! 

use std::marker::PhantomData;

/// Basic discrete constraint trait
pub trait DiscreteConstraint: Clone {
    /// the type of action
    type Action;
    /// the type of (action) value which the mask is applied to
    type Value;
    /// apply the mask and return the result
    fn apply(self, values: Self::Value) -> Self::Value;
    /// compute the mean, considering the constraints
    fn mean_dim(self, dim: usize, values: Self::Value) -> Self::Value;
    /// return the number of possible actions
    fn n_possible_actions(&self) -> Option<usize>;
    /// checks if given action is possible
    fn is_possible(&self, action: Self::Action) -> bool;
}

/// global unconstrained constraint type
#[derive(Debug, Clone)]
pub struct Unconstrained<A: Clone, V: Clone> { _p: PhantomData<(A, V)>}
impl<A: Clone, V: Clone> DiscreteConstraint for Unconstrained<A, V> {
    type Action = A;
    type Value = V;
    fn apply(self, values: Self::Value) -> Self::Value { values }
    fn mean_dim(self, _: usize, values: Self::Value) -> Self::Value { values }
    fn n_possible_actions(&self) -> Option<usize> { None }
    fn is_possible(&self, _: Self::Action) -> bool { true }
}