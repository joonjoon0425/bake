use rand::random_range;

#[derive(Debug, Clone)]
pub struct SumTree {
    tree: Vec<f32>,
    n: usize,
}

impl SumTree {
    pub fn new(capacity: usize) -> Self {
        let depth = (capacity as f32).log2().ceil();
        let n = depth.exp2() as usize;
        let tree = vec![0f32; 2 * n];
        Self { tree, n }
    }

    pub fn from_vec(vec: Vec<f32>) -> Self {
        let capacity = vec.len();
        let depth = (capacity as f32).log2().ceil();
        let n = depth.exp2() as usize;
        let mut tree = vec![0f32; 2 * n];
        for i in 0..capacity {
            tree[n + i] = vec[i];
        }
        let mut index = n / 2;
        while index >= 1 {
            let mut tmp = index;
            let r = index * 2 - 1;
            while tmp <= r {
                tree[tmp] = tree[2 * tmp] + tree[2 * tmp + 1];
                tmp += 1;
            }
            index /= 2;
        }

        Self { tree, n }
    }

    pub fn update(&mut self, index: usize, val: f32) {
        let mut index = index + self.n;
        self.tree[index] = val;

        index /= 2;
        while index >= 1 {
            self.tree[index] = self.tree[2 * index] + self.tree[2 * index + 1];
            index /= 2;
        }
    }

    pub fn get_total(&self) -> f32 {
        self.tree[1]
    }

    pub fn sample_idx(&self, n: usize) -> Vec<f32> {
        let range = self.get_total() / (n as f32);
        let mut vec = vec![0f32; n];
        for i in 0..n {
            let mut r = random_range(i as f32 * range..(i + 1) as f32 * range);
            let mut index = 1;
            while index < self.n {
                if self.tree[2 * index] >= r {
                    index = 2 * index;
                } else {
                    r -= self.tree[2 * index];
                    index = 2 * index + 1;
                }
            }
            vec[i] = self.tree[index];//index - self.n;
        }

        vec
    }

    pub fn is_correct(&self) {
        let mut index = self.n / 2;
        while index >= 1 {
            let mut tmp = index;
            let r = index * 2 - 1;
            while tmp <= r {
                if self.tree[tmp] != self.tree[2 * tmp] + self.tree[2 * tmp + 1] {
                    panic!("Wrong sum on index {tmp}");
                }
                tmp += 1;
            }
            index /= 2;
        }
    }
}