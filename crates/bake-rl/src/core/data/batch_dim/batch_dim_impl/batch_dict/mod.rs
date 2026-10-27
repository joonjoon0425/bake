//! Dictionary which can contain any type of `WithBatchDim` types

use std::{any::Any, collections::BTreeMap, range::Range};
use crate::core::data::BatchShape;

/// Dictionary with batch dimension
#[derive(Debug, Clone)]
pub struct BatchDict {
    dict: BTreeMap<String, BatchDictLeaf>
}

pub type Erased = Box<dyn Any + Sync + Send>;
/// the leaf struct of `BatchDict`
#[derive(Debug)]
pub struct BatchDictLeaf {
    value: Erased,
    type_name: &'static str,
    vtable: WithBatchDimOpsVTable,
}

/// handmade vtable for functions of `WithBatchDim`
#[derive(Debug, Clone)]
struct WithBatchDimOpsVTable {
    batch_shape: fn(&Erased) -> BatchShape,
    cat: fn(Vec<Erased>, isize) -> Erased,
    stack: fn(Vec<Erased>) -> Erased,
    unstack: fn(Erased) -> Vec<Erased>,
    reshape: fn(Erased, BatchShape) -> Erased,
    squeeze: fn(Erased, isize) -> Erased,
    unsqueeze: fn(Erased, isize) -> Erased,
    slice: fn(Erased, isize, Range<usize>) -> Erased,
    select: fn(Erased, isize, burn::Tensor<1, burn::prelude::Int>) -> Erased,
    assign_inplace: fn(&mut Erased, Erased, usize),
    zeros_like: fn(usize, &Erased) -> Erased,
    clone: fn(&Erased) -> Erased,
}

mod dict;
mod vtable;

// unit tests written by Claude

#[cfg(test)]
mod tests {
    //! Unit tests for `BatchDict`
    //! - put this file at `batch_dict/tests.rs` and add `#[cfg(test)] mod tests;` to `batch_dict/mod.rs`
    //! - tests marked with `// [EXPECTED FAIL @50641f2]` describe the intended behavior and fail on commit 50641f2

    use burn::prelude::*;
    use ndarray::{ArrayD, IxDyn};

    use crate::core::data::batch_dim::convert::DetachBatchDim;
    use crate::core::data::{BatchArray, BatchDict, BatchEmpty, BatchShape, BatchTensor, WithBatchDim};

    // ───────────── helpers ─────────────

    fn d() -> Device { Device::default() }

    /// one element (empty batch) whose features are `[id, id + 0.5]`; inner tensor `[1, 2]`
    fn elem_t(id: f32) -> BatchTensor<2> {
        BatchTensor::with_empty_batch_dim(Tensor::from_floats([[id, id + 0.5]], &d()))
    }

    /// one element (empty batch) holding a single string
    fn elem_s(s: &str) -> BatchArray<String> {
        BatchArray::with_empty_batch_dim(ArrayD::from_elem(IxDyn(&[]), s.to_string()))
    }

    /// one element (empty batch) of a dict with keys `obs`, `mask`, `text`
    fn elem_dict(id: f32) -> BatchDict {
        let mut dict = BatchDict::new();
        dict.insert("obs", elem_t(id)).unwrap();
        dict.insert("mask", BatchEmpty::<()>::with_empty_batch_dim()).unwrap();
        dict.insert("text", elem_s(&format!("s{id}"))).unwrap();
        dict
    }

    /// dict of batch `[n]` where element i has id i
    fn batch_dict(n: usize) -> BatchDict {
        BatchDict::stack((0..n).map(|i| elem_dict(i as f32)).collect())
    }

    /// the ids (first feature) of `obs`, in batch order
    fn obs_ids(dict: &BatchDict) -> Vec<f32> {
        let (t, _) = dict.get::<BatchTensor<2>>("obs").unwrap().clone().detach_batch_dim();
        let v: Vec<f32> = t.into_data().try_into_vec_as().unwrap();
        v.chunks(2).map(|c| c[0]).collect()
    }

    /// the strings of `text`, in batch order
    fn texts(dict: &BatchDict) -> Vec<String> {
        let (a, _) = dict.get::<BatchArray<String>>("text").unwrap().clone().detach_batch_dim();
        a.iter().cloned().collect()
    }

    fn idx(v: &[i64]) -> Tensor<1, Int> { Tensor::from_ints(v, &d()) }

    // ───────────── insert / get / get_mut / remove ─────────────

    #[test]
    fn insert_and_get_roundtrip() {
        let dict = elem_dict(3.0);
        assert_eq!(obs_ids(&dict), vec![3.0]);
        assert_eq!(texts(&dict), vec!["s3".to_string()]);
        assert!(dict.get::<BatchEmpty<()>>("mask").is_ok());
    }

    #[test]
    fn insert_overwrites_existing_key() {
        let mut dict = elem_dict(1.0);
        dict.insert("obs", elem_t(7.0)).unwrap();
        assert_eq!(obs_ids(&dict), vec![7.0]);
    }

    #[test]
    fn insert_overwrite_with_different_type() {
        let mut dict = elem_dict(1.0);
        dict.insert("obs", elem_s("now a string")).unwrap();
        assert!(dict.get::<BatchTensor<2>>("obs").is_err());
        assert!(dict.get::<BatchArray<String>>("obs").is_ok());
    }

    #[test]
    fn get_missing_key_lists_existing_keys() {
        let dict = elem_dict(0.0);
        let err = dict.get::<BatchTensor<2>>("obz").unwrap_err();
        assert!(err.contains("obz"), "{err}");
        for k in ["obs", "mask", "text"] {
            assert!(err.contains(k), "missing key `{k}` in: {err}");
        }
    }

    #[test]
    fn get_wrong_type_reports_both_types() {
        let dict = elem_dict(0.0);
        let err = dict.get::<BatchTensor<3>>("obs").unwrap_err();
        assert!(err.contains("BatchTensor<2"), "{err}");
        assert!(err.contains("BatchTensor<3"), "{err}");
    }

    #[test]
    fn get_mut_changes_value() {
        let mut dict = elem_dict(0.0);
        *dict.get_mut::<BatchTensor<2>>("obs").unwrap() = elem_t(5.0);
        assert_eq!(obs_ids(&dict), vec![5.0]);
    }

    #[test]
    fn get_mut_wrong_type_is_err() {
        let mut dict = elem_dict(0.0);
        assert!(dict.get_mut::<BatchEmpty<()>>("obs").is_err());
    }

    #[test]
    fn remove_returns_value_and_deletes_key() {
        let mut dict = elem_dict(2.0);
        let t = dict.remove::<BatchTensor<2>>("obs").unwrap();
        assert_eq!(t.batch_shape(), Vec::<usize>::new());
        assert!(dict.get::<BatchTensor<2>>("obs").is_err());
        assert!(dict.get::<BatchArray<String>>("text").is_ok());
    }

    #[test]
    fn remove_missing_key_is_err() {
        let mut dict = elem_dict(0.0);
        assert!(dict.remove::<BatchTensor<2>>("nope").is_err());
    }

    // [EXPECTED FAIL @50641f2] the value is removed from the map before the type check, so it is lost
    #[test]
    fn remove_wrong_type_keeps_value() {
        let mut dict = elem_dict(4.0);
        assert!(dict.remove::<BatchEmpty<()>>("obs").is_err());
        assert_eq!(obs_ids(&dict), vec![4.0]);
    }

    #[test]
    fn clone_is_deep() {
        let dict = elem_dict(1.0);
        let mut cloned = dict.clone();
        *cloned.get_mut::<BatchTensor<2>>("obs").unwrap() = elem_t(9.0);
        assert_eq!(obs_ids(&dict), vec![1.0]);
        assert_eq!(obs_ids(&cloned), vec![9.0]);
    }

    // ───────────── batch_shape ─────────────

    #[test]
    fn batch_shape_of_empty_dict_is_empty() {
        assert_eq!(BatchDict::new().batch_shape(), Vec::<usize>::new());
    }

    #[test]
    fn batch_shape_of_element_is_empty() {
        assert_eq!(elem_dict(0.0).batch_shape(), Vec::<usize>::new());
    }

    // ───────────── stack / unstack / cat ─────────────

    #[test]
    fn stack_builds_batch_for_every_key() {
        let dict = batch_dict(3);
        assert_eq!(dict.batch_shape(), vec![3]);
        assert_eq!(dict.get::<BatchTensor<2>>("obs").unwrap().batch_shape(), vec![3]);
        assert_eq!(dict.get::<BatchEmpty<()>>("mask").unwrap().batch_shape(), vec![3]);
        assert_eq!(dict.get::<BatchArray<String>>("text").unwrap().batch_shape(), vec![3]);
        assert_eq!(obs_ids(&dict), vec![0.0, 1.0, 2.0]);
        assert_eq!(texts(&dict), vec!["s0", "s1", "s2"]);
    }

    #[test]
    fn unstack_is_inverse_of_stack() {
        let parts = batch_dict(3).unstack();
        assert_eq!(parts.len(), 3);
        for (i, p) in parts.iter().enumerate() {
            assert_eq!(p.batch_shape(), Vec::<usize>::new());
            assert_eq!(obs_ids(p), vec![i as f32]);
            assert_eq!(texts(p), vec![format!("s{}", i as f32)]);
        }
    }

    #[test]
    #[should_panic]
    fn stack_with_missing_key_panics() {
        let mut b = elem_dict(1.0);
        b.remove::<BatchArray<String>>("text").unwrap();
        BatchDict::stack(vec![elem_dict(0.0), b]);
    }

    // [EXPECTED FAIL @50641f2] the message prints the type name instead of the key name
    #[test]
    #[should_panic]
    fn stack_missing_key_message_names_the_key() {
        let mut b = elem_dict(1.0);
        b.remove::<BatchArray<String>>("text").unwrap();
        BatchDict::stack(vec![elem_dict(0.0), b]);
    }

    #[test]
    #[should_panic]
    fn stack_same_key_different_type_panics() {
        let mut b = elem_dict(1.0);
        b.insert("obs", elem_s("wrong")).unwrap();
        BatchDict::stack(vec![elem_dict(0.0), b]);
    }

    #[test]
    fn cat_along_dim0() {
        let dict = BatchDict::cat(vec![batch_dict(2), batch_dict(3)], 0);
        assert_eq!(dict.batch_shape(), vec![5]);
        assert_eq!(obs_ids(&dict), vec![0.0, 1.0, 0.0, 1.0, 2.0]);
        assert_eq!(texts(&dict), vec!["s0", "s1", "s0", "s1", "s2"]);
    }

    // [EXPECTED FAIL @50641f2] the message says `BatchDict::stack` and prints the type name instead of the key
    #[test]
    #[should_panic]
    fn cat_missing_key_message_names_cat_and_key() {
        let mut b = batch_dict(2);
        b.remove::<BatchArray<String>>("text").unwrap();
        BatchDict::cat(vec![batch_dict(2), b], 0);
    }

    // ───────────── reshape / squeeze / unsqueeze / slice ─────────────

    #[test]
    fn reshape_applies_to_every_key() {
        let dict = batch_dict(6).reshape(vec![2, 3]);
        assert_eq!(dict.batch_shape(), vec![2, 3]);
        assert_eq!(dict.get::<BatchEmpty<()>>("mask").unwrap().batch_shape(), vec![2, 3]);
        assert_eq!(dict.get::<BatchArray<String>>("text").unwrap().batch_shape(), vec![2, 3]);
        assert_eq!(obs_ids(&dict), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn unsqueeze_then_squeeze() {
        let dict = batch_dict(3).unsqueeze(0);
        assert_eq!(dict.batch_shape(), vec![1, 3]);
        assert_eq!(dict.get::<BatchArray<String>>("text").unwrap().batch_shape(), vec![1, 3]);
        let dict = dict.squeeze(0);
        assert_eq!(dict.batch_shape(), vec![3]);
        assert_eq!(texts(&dict), vec!["s0", "s1", "s2"]);
    }

    #[test]
    fn slice_dim0() {
        let dict = batch_dict(5).slice(0, (1..4).into());
        assert_eq!(dict.batch_shape(), vec![3]);
        assert_eq!(obs_ids(&dict), vec![1.0, 2.0, 3.0]);
        assert_eq!(texts(&dict), vec!["s1", "s2", "s3"]);
    }

    #[test]
    fn slice_dim1_after_reshape() {
        let dict = batch_dict(6).reshape(vec![2, 3]).slice(1, (1..3).into());
        assert_eq!(dict.batch_shape(), vec![2, 2]);
        assert_eq!(obs_ids(&dict), vec![1.0, 2.0, 4.0, 5.0]);
    }

    // ───────────── select ─────────────

    // [EXPECTED FAIL @50641f2] `check_select` uses `idx.dims().len()` (rank) instead of `idx.dims()[0]`
    #[test]
    fn select_dim0_with_duplicates() {
        let dict = batch_dict(4).select(0, idx(&[3, 0, 3]));
        assert_eq!(dict.batch_shape(), vec![3]);
        assert_eq!(dict.get::<BatchEmpty<()>>("mask").unwrap().batch_shape(), vec![3]);
        assert_eq!(obs_ids(&dict), vec![3.0, 0.0, 3.0]);
        assert_eq!(texts(&dict), vec!["s3", "s0", "s3"]);
    }

    // [EXPECTED FAIL @50641f2] same cause as above
    #[test]
    fn select_dim1_after_reshape() {
        let dict = batch_dict(6).reshape(vec![2, 3]).select(1, idx(&[2, 0]));
        assert_eq!(dict.batch_shape(), vec![2, 2]);
        assert_eq!(obs_ids(&dict), vec![2.0, 0.0, 5.0, 3.0]);
        assert_eq!(texts(&dict), vec!["s2", "s0", "s5", "s3"]);
    }

    // ───────────── zeros_like / assign_inplace ─────────────

    /// `zeros_like(capacity, template)` prepends `capacity` to the template's batch shape
    #[test]
    fn zeros_like_keeps_keys_and_capacity() {
        let buf = BatchDict::zeros_like(5, &elem_dict(0.0));
        assert_eq!(buf.batch_shape(), vec![5]);
        assert_eq!(buf.get::<BatchArray<String>>("text").unwrap().batch_shape(), vec![5]);
        assert_eq!(buf.get::<BatchEmpty<()>>("mask").unwrap().batch_shape(), vec![5]);
    }

    #[test]
    fn assign_inplace_writes_rows() {
        let mut buf = BatchDict::zeros_like(5, &elem_dict(0.0));
        buf.assign_inplace(batch_dict(2), 3);
        let ids = obs_ids(&buf);
        assert_eq!(&ids[3..5], &[0.0, 1.0]);
        assert_eq!(&texts(&buf)[3..5], &["s0".to_string(), "s1".to_string()]);
    }

    // [EXPECTED FAIL @50641f2] `zip` stops at the shorter dict, so the missing key is silently skipped
    #[test]
    #[should_panic]
    fn assign_inplace_with_missing_key_panics() {
        let mut buf = BatchDict::zeros_like(5, &elem_dict(0.0));
        let mut src = batch_dict(2);
        src.remove::<BatchArray<String>>("text").unwrap();
        buf.assign_inplace(src, 0);
    }

    #[test]
    #[should_panic]
    fn assign_inplace_with_different_key_panics() {
        let mut buf = BatchDict::zeros_like(5, &elem_dict(0.0));
        let mut src = batch_dict(2);
        let t = src.remove::<BatchArray<String>>("text").unwrap();
        src.insert("txt", t).unwrap();
        buf.assign_inplace(src, 0);
    }

    // ───────────── nesting ─────────────

    #[test]
    fn nested_dict_follows_outer_ops() {
        let make = |i: f32| {
            let mut outer = BatchDict::new();
            outer.insert("inner", elem_dict(i)).unwrap();
            outer.insert("reward", BatchTensor::<1>::with_empty_batch_dim(Tensor::from_floats([i], &d()))).unwrap();
            outer
        };
        let dict = BatchDict::stack((0..4).map(|i| make(i as f32)).collect()).slice(0, (1..3).into());
        assert_eq!(dict.batch_shape(), vec![2]);
        let inner = dict.get::<BatchDict>("inner").unwrap();
        assert_eq!(inner.batch_shape(), vec![2]);
        assert_eq!(obs_ids(inner), vec![1.0, 2.0]);
    }

    #[test]
    fn shape_type_is_vec() {
        // guard so that `BatchShape` stays `Vec<usize>` (the tests above compare with `vec![..]`)
        let s: BatchShape = vec![1, 2];
        assert_eq!(s.len(), 2);
    }
}