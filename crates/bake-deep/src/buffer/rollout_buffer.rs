use burn::{Tensor, tensor::Device};

use crate::types::{ActionMask, Batch, Batchable, NoMask, Transition};

/// Rollout Buffer Implementation. Also works as Episode Buffer if n = None is given
pub struct RolloutBuffer<Obs: Batchable, Action: Batchable, Mask: ActionMask + Batchable = NoMask, Extra: Batchable = ()>{
    obss: Vec<Obs>,
    actions: Vec<Action>,
    rewards: Vec<f32>,
    next_obss: Vec<Obs>,
    terminated: Vec<f32>,
    truncated: Vec<f32>,

    masks: Vec<Mask>,
    next_masks: Vec<Mask>,
    extras: Vec<Extra>,

    device: Device,

    n: Option<usize>,
}

impl<Obs: Batchable, Action: Batchable, Mask: ActionMask + Batchable, Extra: Batchable> RolloutBuffer<Obs, Action, Mask, Extra> {
    pub fn new(n: Option<usize>, device: Device) -> Self {
        Self {
            obss: vec![],
            actions: vec![],
            rewards: vec![],
            next_obss: vec![],
            terminated: vec![],
            truncated: vec![],
            masks: vec![],
            next_masks: vec![],
            extras: vec![],

            device,
            n,
        }
    }

    pub fn len(&self) -> usize {
        self.obss.len()
    }

    pub fn is_full(&self) -> bool {
        let len = self.len();
        if let Some(n) = self.n {
            return len == n;
        } else {
            return self.terminated[len - 1] == 1f32 || self.truncated[len - 1] == 1f32;
        }
    }

    pub fn push(&mut self, t: Transition<Obs, Action, Mask, Extra>) {
        self.obss.push(t.obs);
        self.actions.push(t.action);
        self.rewards.push(t.reward);
        self.next_obss.push(t.next_obs);
        self.terminated.push(if t.terminated { 1f32 } else { 0f32 });
        self.truncated.push(if t.truncated { 1f32 } else { 0f32 });
        self.masks.push(t.mask);
        self.next_masks.push(t.next_mask);
        self.extras.push(t.extra);
    }

    pub fn pop(&mut self) -> Batch<Obs, Action, Mask, Extra> {
        let batched_steps = Batch {
            obss: Obs::batch(self.obss.clone(), &self.device),
            actions: Action::batch(self.actions.clone(), &self.device),
            rewards: Tensor::from_floats(self.rewards.as_slice(), &self.device),
            next_obss: Obs::batch(self.next_obss.clone(), &self.device),
            terminated: Tensor::from_floats(self.terminated.as_slice(), &self.device),
            truncated: Tensor::from_floats(self.truncated.as_slice(), &self.device),
            masks: Mask::batch(self.masks.clone(), &self.device),
            next_masks: Mask::batch(self.next_masks.clone(), &self.device),
            extras: Extra::batch(self.extras.clone(), &self.device),

            batch_size: self.len()
        };

        self.obss.clear();
        self.actions.clear();
        self.rewards.clear();
        self.next_obss.clear();
        self.terminated.clear();
        self.truncated.clear();
        self.masks.clear();
        self.next_masks.clear();
        self.extras.clear();

        batched_steps
    }
}