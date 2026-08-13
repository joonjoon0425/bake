pub trait Mask {
    fn is_possible(&self, action: usize) -> bool;
    fn possible_actions(&self) -> impl Iterator<Item = usize> + '_;
    fn n_possible_actions(&self) -> usize;
    fn n_actions(&self) -> usize;
}
#[derive(Debug, Clone, Copy)]
pub struct DiscreteMask<const D: usize>([bool; D]);

#[derive(Debug, Clone, Copy)]
pub struct NoMask<const D: usize>;

impl<const D: usize> Mask for DiscreteMask<D> {
    fn is_possible(&self, action: usize) -> bool {
        self.0[action]
    }

    fn possible_actions(&self) -> impl Iterator<Item = usize> + '_ {
        self.0.iter().enumerate()
            .filter(|(_, possible)| **possible )
            .map(|(action, _)| action )
    }

    fn n_possible_actions(&self) -> usize {
        self.0.iter().filter(|possible| **possible ).count()
    }

    fn n_actions(&self) -> usize {
        D
    }
}

impl<const D: usize> Mask for NoMask<D> {
    fn is_possible(&self, _: usize) -> bool { true }

    fn possible_actions(&self) -> impl Iterator<Item = usize> {
        0..D
    }

    fn n_possible_actions(&self) -> usize {
        D
    }

    fn n_actions(&self) -> usize {
        D
    }
}