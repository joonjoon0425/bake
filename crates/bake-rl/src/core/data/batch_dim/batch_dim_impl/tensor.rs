use burn::{prelude::*, tensor::kind::Basic};

use crate::core::data::batch_dim::{BatchShape, WithBatchDim, check_batch_dim, check_batch_dim_except, n_batch, resolve_negative_index};

/// `burn::Tensor`, with batch dimension
#[derive(Debug, Clone)]
pub struct BatchTensor<const D: usize, K: Basic = Float> {
    shape: BatchShape,
    t: Tensor<D, K>,
}

impl<const D: usize, K: Basic> BatchTensor<D, K> {
    /// create a new `BatchTensor` with empty batch dimension, from given tensor
    pub fn new(tensor: Tensor<D, K>) -> Self {
        debug_assert_eq!(tensor.dims()[0], 1, "empty batch shape requires dim 0 == 1");

        Self {
            shape: vec![],
            t: tensor
        }
    }
}

impl<const D: usize, K: Basic> WithBatchDim for BatchTensor<D, K> {
    fn batch_shape(&self) -> &BatchShape {
        &self.shape
    }

    fn reshape(self, shape: BatchShape) -> Self {
        let m = self.shape.into_iter().product::<usize>();
        let n = shape.iter().product::<usize>();

        if m != n { panic!("Invalid shape was given for reshape operation") }

        Self {
            shape,
            ..self
        }
    }

    fn squeeze(mut self, dim: isize) -> Self {
        let dim = resolve_negative_index(self.shape.len(), dim);

        if self.shape[dim] != 1 { panic!("The dimension is not 1") }
        self.shape.remove(dim);
        self
    }

    fn unsqueeze(mut self, dim: isize) -> Self {
        let dim = resolve_negative_index(self.shape.len() + 1, dim);

        self.shape.insert(dim, 1);

        self
    }

    fn assign_inplace(&mut self, data: Self, idx: usize) {
        assert_eq!(&data.shape[1..], &self.shape[1..], "batch dims except axis 0 must match");
        assert!(idx + data.shape[0] <= self.shape[0], "out of range");
        
        let row = n_batch(&self.shape[1..]);
        let k = data.shape[0];
        self.t.inplace(|a| a.slice_assign((idx * row)..(idx + k) * row, data.t));
    }

    fn stack(dat: Vec<Self>) -> Self {
        assert!(!dat.is_empty(), "Given data vector is empty");
        assert!(check_batch_dim(&dat), "The batch dimension does not matches");

        let n = dat.len();
        let mut shape = dat[0].shape.clone();
        shape.insert(0, n);
        let inner = dat.into_iter().map(|d| d.t).collect();
        let stacked = Tensor::cat(inner, 0);
        Self {
            shape,
            t: stacked
        }
    }

    fn cat(dat: Vec<Self>, dim: isize) -> Self {
        assert!(!dat.is_empty(), "The given data is empty");

        let mut shape = dat[0].shape.clone();
        let mut dims = dat[0].t.dims();
        let n_features: usize = dims[1..].iter().product();

        let dim = resolve_negative_index(shape.len(), dim);
        assert!(check_batch_dim_except(&dat, dim), "The batch dimension does not matches");

        // fold
        let out = n_batch(&shape[..dim]);
        let res = n_batch(&shape[dim + 1..]) * n_features;
        let mut k_final = 0;

        let t_vec = dat.into_iter().map(
            |d| {
                let k = d.shape[dim];
                k_final += k;
                d.t.reshape([out, k, res])
            }
        ).collect();

        shape[dim] = k_final;
        dims[0] = n_batch(&shape);
        let concatanated = Tensor::cat(t_vec, 1).reshape(dims);
        
        Self {
            shape,
            t: concatanated
        }
    }

    fn slice(self, dim: isize, range: std::range::Range<usize>) -> Self {
        let mut shape = self.shape.clone();
        let dim = resolve_negative_index(shape.len(), dim);
        let mut dims = self.t.dims();
        let n_features: usize = dims[1..].iter().product();

        let out = n_batch(&shape[..dim]);
        let k = shape[dim];
        let res = n_batch(&shape[dim + 1..]) * n_features;

        let k_final = range.iter().len();
        shape[dim] = k_final;
        dims[0] = n_batch(&shape);
        let t = self.t
            .reshape([out, k , res])
            .narrow(1, range.start, k_final)
            .reshape(dims);

        Self {
            shape,
            t
        }
    }

    fn select(self, dim: isize, idx: Tensor<1, Int>) -> Self {
        let mut shape = self.shape.clone();
        let dim = resolve_negative_index(shape.len(), dim);
        let mut dims = self.t.dims();
        let n_features: usize = dims[1..].iter().product();

        let out = n_batch(&shape[..dim]);
        let k = shape[dim];
        let res = n_batch(&shape[dim + 1..]) * n_features;

        let k_final = idx.dims()[0];
        shape[dim] = k_final;
        dims[0] = n_batch(&shape);

        let t = self.t
            .reshape([out, k , res])
            .select(1, idx)
            .reshape(dims);

        Self {
            shape,
            t
        }
    }

    fn zeros_like(capacity: usize, data: &Self) -> Self {
        let mut shape = data.t.shape();
        shape[0] = shape[0] * capacity;

        let t = Tensor::zeros(shape, &data.t.device());

        let mut shape = data.shape.clone();
        shape.insert(0, capacity);

        Self {
            shape,
            t
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::prelude::*;
    use burn::tensor::kind::Basic;
    use crate::core::data::batch_dim::{BatchShape, BatchTensor, n_batch, WithBatchDim};
    fn d() -> Device { Device::default() }

    fn bt<const D: usize, K: Basic>(shape: BatchShape) -> BatchTensor<D, K> {
        let mut s = [2; D];
        let n = n_batch(&shape);
        s[0] = n;
        let t = Tensor::ones(s, &d());
        BatchTensor {
            shape,
            t
        }
    }

    #[test]
    fn reshape() {
        let bt = bt::<2, Float>(vec![2, 3]);
        let bt = bt.reshape(vec![3, 2]);
        assert!(vec![3, 2] == *bt.batch_shape(), "The reshape did not worked as expected");
        let bt = bt.reshape(vec![1, 6]);
        assert!(vec![1, 6] == *bt.batch_shape(), "The reshape did not worked as expected");
    }

    #[test]
    #[should_panic]
    fn reshape_panic() {
        let bt = bt::<3, Float>(vec![3, 4, 2]);
        bt.reshape(vec![6, 2]);
    }

    #[test]
    fn squeeze() {
        let bt = bt::<4, Float>(vec![2, 3, 4, 1]);
        let bt = bt.squeeze(3);
        assert!(vec![2, 3, 4] == *bt.batch_shape(), "squeeze did not worked as expected");
    }

    #[test]
    fn squeeze_neg_idx() {
        let bt = bt::<3, Float>(vec![2, 1, 2]);
        let bt = bt.squeeze(-2);
        assert!(vec![2, 2] == *bt.batch_shape(), "squeeze did not worked as expected with negative indexing");
    }

    #[test]
    #[should_panic]
    fn squeeze_panic() {
        let bt = bt::<3, Float>(vec![2, 1, 2]);
        bt.squeeze(0);
    }

    #[test]
    fn unsqueeze() {
        let bt = bt::<2, Float>(vec![2, 5]);
        let bt = bt.unsqueeze(0);
        assert!(vec![1, 2, 5] == *bt.batch_shape(), "unsqueeze did not work as expected");
    }

    #[test]
    fn unsqueeze_neg_idx() {
        let bt = bt::<3, Float>(vec![2, 5, 2]);
        let bt = bt.unsqueeze(-2);
        println!("after unsqueeze: {:?}", bt.batch_shape());
        assert!(vec![2, 5, 1, 2] == *bt.batch_shape(), "unsqueeze did not work as expected with negative indexing");
    }

    #[test]
    #[should_panic]
    fn unsqueeze_panic() {
        let bt = bt::<3, Float>(vec![2, 5, 2]);
        bt.unsqueeze(4);
    }

    #[test]
    fn stack() {
        let bts = vec![bt::<2, Float>(vec![2, 3]); 3];
        let stacked = BatchTensor::stack(bts);
        assert!(vec![3, 2, 3] == *stacked.batch_shape(), "stack did not work as expected");
    }

    #[test]
    #[should_panic]
    fn stack_panic() {
        let bts = vec![
            bt::<2, Float>(vec![2, 1]),
            bt::<2, Float>(vec![2, 2])
        ];
        BatchTensor::stack(bts);
    }

    #[test]
    fn cat() {
        let bts = vec![
            bt::<4, Float>(vec![3, 1, 2, 3]),
            bt::<4, Float>(vec![3, 4, 2, 3])
        ];
        let cat = BatchTensor::cat(bts, 1);
        assert!(vec![3, 5, 2, 3] == *cat.batch_shape(), "cat did not work as expected");
    }

    #[test]
    #[should_panic]
    fn cat_panic() {
        let bts = vec![
            bt::<4, Float>(vec![3, 2, 1, 2]),
            bt::<4, Float>(vec![3, 1, 2, 1])
        ];
        BatchTensor::cat(bts, 0);
    }

    // tests by Claude

    // ───────────── 값 검사용 보조 함수 ─────────────

    /// 배치 원소 i의 특징이 [i, i + 0.5]인 `[n, 2]` 텐서 (n = 배치 원소 수)
    fn seq(shape: BatchShape) -> BatchTensor<2, Float> {
        let n = n_batch(&shape);
        let v: Vec<f32> = (0..n).flat_map(|i| [i as f32, i as f32 + 0.5]).collect();
        BatchTensor { shape, t: Tensor::from_data(TensorData::new(v, [n, 2]), &d()) }
    }

    /// 각 배치 원소가 원래 몇 번째 원소였는지 (첫 특징값), 특징 두 번째 값이 +0.5인지도 확인
    fn ids(b: &BatchTensor<2, Float>) -> Vec<usize> {
        let v: Vec<f32> = b.t.clone().into_data().try_to_vec::<f32>().unwrap();
        v.chunks(2)
            .map(|c| {
                assert_eq!(c[1], c[0] + 0.5, "feature row was broken");
                c[0] as usize
            })
            .collect()
    }

    /// 불변식: 텐서 0번 축 길이 == 배치 원소 수
    fn check_inv<const D: usize, K: Basic>(b: &BatchTensor<D, K>) {
        assert_eq!(b.t.dims()[0], n_batch(&b.shape), "invariant broken: dims[0] != n_batch");
    }

    // ───────────── new / 메타데이터 ─────────────

    #[test]
    fn new_empty_batch() {
        let b = BatchTensor::new(Tensor::<2>::ones([1, 3], &d()));
        assert!(b.batch_shape().is_empty());
        check_inv(&b);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn new_panic_dim0_not_1() {
        BatchTensor::new(Tensor::<2>::ones([3, 3], &d()));
    }

    #[test]
    fn unsqueeze_empty_batch() {
        let b = BatchTensor::new(Tensor::<2>::ones([1, 3], &d())).unsqueeze(0);
        assert_eq!(vec![1], *b.batch_shape());
        check_inv(&b);
    }

    #[test]
    fn unsqueeze_last() {
        let b = bt::<2, Float>(vec![2, 3]).unsqueeze(-1);
        assert_eq!(vec![2, 3, 1], *b.batch_shape());
        check_inv(&b);
    }

    #[test]
    #[should_panic]
    fn squeeze_empty_batch_panic() {
        // n_dims = 0 에서 음수 인덱스: 무한 루프가 아니라 패닉이어야 함
        BatchTensor::new(Tensor::<2>::ones([1, 3], &d())).squeeze(-1);
    }

    #[test]
    #[should_panic]
    fn negative_index_out_of_range_panic() {
        bt::<2, Float>(vec![2, 3]).squeeze(-3);
    }

    #[test]
    fn reshape_keeps_values() {
        let b = seq(vec![2, 3]).reshape(vec![6]);
        assert_eq!(vec![6], *b.batch_shape());
        assert_eq!(vec![0, 1, 2, 3, 4, 5], ids(&b));
    }

    // ───────────── stack ─────────────

    #[test]
    fn stack_values() {
        let b = BatchTensor::stack(vec![seq(vec![2]), seq(vec![2]), seq(vec![2])]);
        assert_eq!(vec![3, 2], *b.batch_shape());
        assert_eq!(vec![0, 1, 0, 1, 0, 1], ids(&b));
        check_inv(&b);
    }

    #[test]
    fn stack_empty_batch() {
        let one = BatchTensor::new(Tensor::<2>::ones([1, 3], &d()));
        let b = BatchTensor::stack(vec![one.clone(), one.clone(), one]);
        assert_eq!(vec![3], *b.batch_shape());
        check_inv(&b);
    }

    #[test]
    #[should_panic]
    fn stack_empty_vec_panic() {
        BatchTensor::<2, Float>::stack(vec![]);
    }

    // ───────────── cat ─────────────

    #[test]
    fn cat_dim0_values() {
        let b = BatchTensor::cat(vec![seq(vec![2]), seq(vec![3])], 0);
        assert_eq!(vec![5], *b.batch_shape());
        assert_eq!(vec![0, 1, 0, 1, 2], ids(&b));
        check_inv(&b);
    }

    #[test]
    fn cat_dim1_values() {
        // [2,1]: (t,n) 원소 = t, [2,2]: 원소 = t*2+n
        let b = BatchTensor::cat(vec![seq(vec![2, 1]), seq(vec![2, 2])], 1);
        assert_eq!(vec![2, 3], *b.batch_shape());
        assert_eq!(vec![0, 0, 1, 1, 2, 3], ids(&b));
        check_inv(&b);
    }

    #[test]
    fn cat_neg_dim() {
        let a = BatchTensor::cat(vec![seq(vec![2, 1]), seq(vec![2, 2])], -1);
        let b = BatchTensor::cat(vec![seq(vec![2, 1]), seq(vec![2, 2])], 1);
        assert_eq!(*a.batch_shape(), *b.batch_shape());
        assert_eq!(ids(&a), ids(&b));
    }

    #[test]
    #[should_panic]
    fn cat_rank_mismatch_panic() {
        BatchTensor::cat(vec![bt::<2, Float>(vec![2, 1]), bt::<2, Float>(vec![2])], 0);
    }

    #[test]
    #[should_panic]
    fn cat_empty_vec_panic() {
        BatchTensor::<2, Float>::cat(vec![], 0);
    }

    // ───────────── slice ─────────────

    #[test]
    fn slice_dim0_values() {
        let b = seq(vec![4]).slice(0, (1..3).into());
        assert_eq!(vec![2], *b.batch_shape());
        assert_eq!(vec![1, 2], ids(&b));
        check_inv(&b);
    }

    #[test]
    fn slice_dim1_values() {
        // [2,3]: (t,n) 원소 = t*3+n
        let b = seq(vec![2, 3]).slice(1, (1..3).into());
        assert_eq!(vec![2, 2], *b.batch_shape());
        assert_eq!(vec![1, 2, 4, 5], ids(&b));
        check_inv(&b);
    }

    #[test]
    fn slice_middle_dim_values() {
        // [2,3,2]: (a,b,c) 원소 = a*6+b*2+c, b축 2..3
        let b = seq(vec![2, 3, 2]).slice(1, (2..3).into());
        assert_eq!(vec![2, 1, 2], *b.batch_shape());
        assert_eq!(vec![4, 5, 10, 11], ids(&b));
        check_inv(&b);
    }

    #[test]
    fn slice_neg_dim() {
        let b = seq(vec![2, 3]).slice(-1, (0..1).into());
        assert_eq!(vec![2, 1], *b.batch_shape());
        assert_eq!(vec![0, 3], ids(&b));
    }

    #[test]
    #[should_panic]
    fn slice_out_of_range_panic() {
        seq(vec![2, 3]).slice(1, (2..5).into());
    }

    // ───────────── select ─────────────

    #[test]
    fn select_dim0_values() {
        let b = seq(vec![2, 3]).select(0, Tensor::from_ints([1], &d()));
        assert_eq!(vec![1, 3], *b.batch_shape());
        assert_eq!(vec![3, 4, 5], ids(&b));
        check_inv(&b);
    }

    #[test]
    fn select_dim1_values() {
        // 순서 바꾸기와 중복 포함
        let b = seq(vec![2, 3]).select(1, Tensor::from_ints([2, 0, 2], &d()));
        assert_eq!(vec![2, 3], *b.batch_shape());
        assert_eq!(vec![2, 0, 2, 5, 3, 5], ids(&b));
        check_inv(&b);
    }

    #[test]
    fn select_neg_dim() {
        let b = seq(vec![2, 3]).select(-1, Tensor::from_ints([1], &d()));
        assert_eq!(vec![2, 1], *b.batch_shape());
        assert_eq!(vec![1, 4], ids(&b));
    }

    #[test]
    fn select_after_flatten() {
        // 리플레이 샘플링 패턴: [T, N] → [T*N] → select(0)
        let b = seq(vec![2, 3]).reshape(vec![6]).select(0, Tensor::from_ints([5, 0], &d()));
        assert_eq!(vec![2], *b.batch_shape());
        assert_eq!(vec![5, 0], ids(&b));
    }

    // ───────────── zeros_like / assign_inplace ─────────────

    #[test]
    fn zeros_like_shape_and_values() {
        let z = BatchTensor::zeros_like(4, &seq(vec![3]));
        assert_eq!(vec![4, 3], *z.batch_shape());
        assert_eq!([12, 2], z.t.dims());
        let v: Vec<f32> = z.t.clone().into_data().try_to_vec::<f32>().unwrap();
        assert!(v.iter().all(|x| *x == 0.0));
        check_inv(&z);
    }

    #[test]
    fn assign_inplace_values() {
        let mut z = BatchTensor::zeros_like(4, &seq(vec![2]));
        z.assign_inplace(seq(vec![2]).unsqueeze(0), 2);
        let v: Vec<f32> = z.t.clone().into_data().try_to_vec::<f32>().unwrap();
        // 4x2 배치 중 [2, *] 행만 채워짐
        assert_eq!(
            v,
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.5, 0.0, 0.0, 0.0, 0.0]
        );
        assert_eq!(vec![4, 2], *z.batch_shape());
    }

    #[test]
    fn assign_inplace_multiple_rows() {
        let mut z = BatchTensor::zeros_like(4, &seq(vec![2]));
        z.assign_inplace(seq(vec![2, 2]), 1);
        let v: Vec<f32> = z.t.clone().into_data().try_to_vec::<f32>().unwrap();
        // 행 1, 2가 원소 0..4로 채워짐
        assert_eq!(&v[4..12], &[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]);
        assert!(v[..4].iter().chain(&v[12..]).all(|x| *x == 0.0));
    }

    #[test]
    #[should_panic]
    fn assign_inplace_out_of_range_panic() {
        let mut z = BatchTensor::zeros_like(4, &seq(vec![2]));
        z.assign_inplace(seq(vec![2, 2]), 3);
    }

    #[test]
    #[should_panic]
    fn assign_inplace_inner_mismatch_panic() {
        let mut z = BatchTensor::zeros_like(4, &seq(vec![2]));
        z.assign_inplace(seq(vec![1, 3]), 0);
    }

    // ───────────── D = 1, Int, Bool ─────────────

    #[test]
    fn rank1_slice_dim1() {
        // 보상 같은 스칼라 필드: 텐서 [6], 배치 [2,3]
        let t: Tensor<1> = Tensor::from_floats([0., 1., 2., 3., 4., 5.], &d());
        let b = BatchTensor { shape: vec![2, 3], t }.slice(1, (1..2).into());
        assert_eq!(vec![2, 1], *b.batch_shape());
        let v: Vec<f32> = b.t.clone().into_data().try_to_vec::<f32>().unwrap();
        assert_eq!(v, vec![1.0, 4.0]);
        check_inv(&b);
    }

    #[test]
    fn int_cat_dim0() {
        let t: Tensor<1, Int> = Tensor::from_ints([7], &d());
        let a = BatchTensor { shape: vec![1], t: t.clone() };
        let b = BatchTensor { shape: vec![1], t };
        let c = BatchTensor::cat(vec![a, b], 0);
        assert_eq!(vec![2], *c.batch_shape());
        // 백엔드에 따라 Int 저장 dtype이 달라서 into_data_as로 읽음
        let v: Vec<i64> = c.t.clone().into_data_as::<i64>().try_to_vec::<i64>().unwrap();
        assert_eq!(v, vec![7, 7]);
    }

    #[test]
    fn bool_stack_select() {
        let t: Tensor<2, Bool> = Tensor::from_bool(TensorData::from([[true, false]]), &d());
        let s = BatchTensor::stack(vec![BatchTensor::new(t.clone()), BatchTensor::new(t)]);
        let s = s.unsqueeze(-1).select(1, Tensor::from_ints([0, 0], &d()));
        assert_eq!(vec![2, 2], *s.batch_shape());
        // 백엔드에 따라 Bool 저장 dtype이 u8일 수 있어서 into_data_as로 읽음
        let v: Vec<bool> = s.t.clone().into_data_as::<bool>().try_to_vec::<bool>().unwrap();
        assert_eq!(v, [true, false].repeat(4));
        check_inv(&s);
    }
}