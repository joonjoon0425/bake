//! The `Tape` struct, which helps making transition tuple (s, a, r, s') and manage environment steps and resets.
//! - Especially useful for vector environments
//! 
use crate::core::env::VectorEnvironment;

/// The `Tape` struct, which helps making transition tuple (s, a, r, s') and manage environment steps and resets.
/// - Especially useful for vector environments
pub struct Tape<E> {
    env: E
}

impl<E: VectorEnvironment> Tape<E> {
    /// create a new tape
    fn new() {}
}