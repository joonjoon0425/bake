#[derive(Debug, Clone, Copy)]
pub struct ActionMask {
    mask: u128,
}

pub struct ActionMaskIterator {
    mask: u128,
}

impl ActionMask {
    pub fn new(mask: u128) -> Self { Self{ mask } }
    pub fn from_n_actions(n_actions: usize) -> Self {
        let mask = (1 << n_actions) - 1;
        Self { mask }
    }

    pub fn n_possible(&self) -> usize {
        self.mask.count_ones() as usize
    }
}

impl Iterator for ActionMaskIterator {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.mask == 0 {
            return None
        } else {
            let ret = self.mask.trailing_zeros() as usize;
            self.mask &= self.mask - 1;
            return Some(ret);
        }
    }
}

impl ExactSizeIterator for ActionMaskIterator {
    fn len(&self) -> usize {
        self.mask.count_ones() as usize
    }
}