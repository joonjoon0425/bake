use std::collections::BTreeMap;
use crate::core::data::{WithBatchDim, batch_dim::batch_dim_impl::batch_dict::{Erased, WithBatchDimOpsVTable}};
use super::{BatchDictLeaf, BatchDict, BatchShape};

impl BatchDict {
    /// create a new BatchDict
    pub fn new() -> Self {
        Self {
            dict: BTreeMap::new()
        }
    }

    /// insert a new key and value; if the given key was already present in dictionary, it replaces with given value and returns the previous value
    pub fn insert<T: WithBatchDim>(&mut self, key: impl Into<String>, value: T) -> Result<Option<BatchDictLeaf>, String> {
        if !self.dict.is_empty() {
            if self.batch_shape() != value.batch_shape() { return Err("BatchDict::insert: The batch shape of dictionary and given value does not matches.".into()) };
        }
        Ok(self.dict.insert(key.into(), BatchDictLeaf {
            value: Box::new(value),
            type_name: std::any::type_name::<T>(),
            vtable: WithBatchDimOpsVTable::new::<T>(),
        }))
    }

    /// get an existing value
    pub fn get<T: WithBatchDim>(&self, key: &str) -> Result<&T, String> {
        let erased = self.dict.get(key).ok_or(
            format!("The item for given key {} does not exists. Following keys exists: {}",
                key,
                {
                    let mut key_type = String::new();
                    for (s, v) in &self.dict {
                        key_type += format!("\n{} as type {}", s, v.type_name).as_str();
                    }
                    key_type
                }
            )
        )?;
        erased.value.downcast_ref::<T>().ok_or(
            format!("The item for given key {} has type {}, but type {} was given",
                key,
                erased.type_name,
                std::any::type_name::<T>()
            )
        )
    }

    /// get an existing value as mutable reference
    pub fn get_mut<T: WithBatchDim>(&mut self, key: &str) -> Result<&mut T, String> {
        // cannot do like `get`. I think it's due to the `{}` block?
        let mut key_type = String::new();
        for (s, v) in self.dict.iter() {
            key_type += format!("\n{} as type {}", s, v.type_name).as_str();
        }

        let erased = self.dict.get_mut(key).ok_or(
            format!("The item for given key {} does not exists. Following keys exists: {}",
                key,
                key_type
            )
        )?;
        erased.value.downcast_mut::<T>().ok_or(
            format!("The item for given key {} has type {}, but type {} was given",
                key,
                erased.type_name,
                std::any::type_name::<T>()
            )
        )
    }

    /// remove the value of given key
    pub fn remove<T: WithBatchDim>(&mut self, key: &str) -> Result<T, String> {
        // check the existence of key, and the type
        self.dict.get(key).ok_or(
            format!("The item for given key {} does not exists. Following keys exists: {}",
                key,
                {
                    let mut key_type = String::new();
                    for (s, v) in &self.dict {
                        key_type += format!("\n{} as type {}", s, v.type_name).as_str();
                    }
                    key_type
                }
            )
        ).and_then(|v| {
            if !v.value.is::<T>() {
                Err(format!("The item for given key {} has type {}, but type {} was given",
                    key,
                    v.type_name,
                    std::any::type_name::<T>()
                ))
            } else {
                Ok(())
            }
        })?;

        let erased = self.dict.remove(key).unwrap();

        Ok(*erased.value.downcast::<T>().unwrap())
    }
}

impl Clone for BatchDictLeaf {
    fn clone(&self) -> Self {
        let value = (self.vtable.clone)(&self.value);
        Self {
            value,
            type_name: self.type_name,
            vtable: self.vtable.clone()
        }
    }
}

impl WithBatchDim for BatchDict {
    fn batch_shape(&self) -> BatchShape {
        for (_, v) in &self.dict {
            return (v.vtable.batch_shape)(&v.value)
        }
        return vec![]    
    }

    fn reshape(self, shape: BatchShape) -> Self {
        let mut dict = BTreeMap::new();
        for (s, v) in self.dict {
            dict.insert(s,  BatchDictLeaf {
                value: (v.vtable.reshape)(v.value, shape.clone()),
                ..v
            });
        }
        Self {
            dict
        }
    }

    fn squeeze(self, dim: isize) -> Self {
        let mut dict = BTreeMap::new();
        for (s, v) in self.dict {
            dict.insert(s, BatchDictLeaf {
                value: (v.vtable.squeeze)(v.value, dim),
                ..v
            });
        }
        Self {
            dict
        }
    }

    fn unsqueeze(self, dim: isize) -> Self {
        let mut dict = BTreeMap::new();
        for (s, v) in self.dict {
            dict.insert(s, BatchDictLeaf {
                value: (v.vtable.unsqueeze)(v.value, dim),
                ..v
            });
        }
        Self {
            dict
        }
    }

    fn cat(dat: Vec<Self>, dim: isize) -> Self {
        // used for checking the total number of values for each keys
        let n = dat.len();
        let mut dict = BTreeMap::new();
        let mut split: BTreeMap<String, (Vec<Erased>, WithBatchDimOpsVTable, &'static str)> = BTreeMap::new();
        
        // collect all elements in each keys
        for d in dat {
            for (s, leaf) in d.dict {
                split.entry(s).or_insert((vec![], leaf.vtable, leaf.type_name)).0.push(leaf.value);
            }
        }

        for (s, (vec, vtable, type_name)) in split {
            assert_eq!(n, vec.len(), "BatchDict::cat: key '{}' present in {} of {} items", s, vec.len(), n);
            let value = (vtable.cat)(vec, dim);
            dict.insert(s, BatchDictLeaf { value, type_name, vtable });
        }

        Self {
            dict
        }
    }
    
    fn stack(dat: Vec<Self>) -> Self {
        // used for checking the total number of values for each keys
        let n = dat.len();
        let mut dict = BTreeMap::new();
        let mut split: BTreeMap<String, (Vec<Erased>, WithBatchDimOpsVTable, &'static str)> = BTreeMap::new();
        
        // collect all elements in each keys
        for d in dat {
            for (s, leaf) in d.dict {
                split.entry(s).or_insert((vec![], leaf.vtable, leaf.type_name)).0.push(leaf.value);
            }
        }

        for (s, (vec, vtable, type_name)) in split {
            assert_eq!(n, vec.len(), "BatchDict::stack: key '{}' present in {} of {} items", s, vec.len(), n);
            let value = (vtable.stack)(vec);
            dict.insert(s, BatchDictLeaf { value, type_name, vtable });
        }

        Self {
            dict
        }
    }

    fn unstack(self) -> Vec<Self> {
        let n = self.batch_shape()[0];
        let mut split: BTreeMap<String, (Vec<Erased>, WithBatchDimOpsVTable, &'static str)> = BTreeMap::new();
        for (s, leaf) in self.dict {
            let vec = (leaf.vtable.unstack)(leaf.value);
            split.insert(s, (vec, leaf.vtable, leaf.type_name));
        }
        let mut dicts = vec![BatchDict { dict: BTreeMap::new() }; n];
        for (s, (vec, vtable, type_name)) in split {
            for (dict, value) in dicts.iter_mut().zip(vec.into_iter()) {
                dict.dict.insert(s.clone(), BatchDictLeaf {
                    value,
                    vtable: vtable.clone(),
                    type_name,
                });
            }
        }

        dicts
    }

    fn slice(self, dim: isize, range: std::range::Range<usize>) -> Self {
        let mut dict = BTreeMap::new();
        for (s, v) in self.dict {
            dict.insert(s,  BatchDictLeaf {
                value: (v.vtable.slice)(v.value, dim, range.clone()),
                ..v
            });
        }
        Self {
            dict
        }
    }

    fn select(self, dim: isize, idx: burn::prelude::Tensor<1, burn::prelude::Int>) -> Self {
        let mut dict = BTreeMap::new();
        for (s, v) in self.dict {
            dict.insert(s,  BatchDictLeaf {
                value: (v.vtable.select)(v.value, dim, idx.clone()),
                ..v
            });
        }
        Self {
            dict
        }
    }

    fn assign_inplace(&mut self, data: Self, idx: usize) {
        assert!(self.dict.iter().len() == data.dict.iter().len(), "BatchDict::assign_inplace: The number of key differs in self and data");
        for ((dst_s, dst_leaf), (src_str, src_leaf)) in self.dict.iter_mut().zip(data.dict.into_iter()) {
            assert!(*dst_s == src_str, "BatchDict::assign_inplace: key mismatch: expected {dst_s}, got {src_str}");
            (dst_leaf.vtable.assign_inplace)(&mut dst_leaf.value, src_leaf.value, idx)
        }
    }

    fn zeros_like(capacity: usize, data: &Self) -> Self {
        let mut dict = BTreeMap::new();
        for (s, leaf) in &data.dict {
            dict.insert(s.clone(), BatchDictLeaf {
                value: (leaf.vtable.zeros_like)(capacity, &leaf.value),
                vtable: leaf.vtable.clone(),
                type_name: leaf.type_name,
            });
        }
        Self {
            dict
        }
    }
}