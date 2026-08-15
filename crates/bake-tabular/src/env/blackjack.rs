//! ## Blackjack
//! This is an implementation of Blackjack game from Sutton & Barto.
//! Got references from https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/toy_text/blackjack.py

use rand::{RngExt, SeedableRng, rngs::StdRng};

use crate::{env::Env, types::NoMask};

const BLACKJACK_CARDS: [usize; 13] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10];

/// A Blackjack game implementation
pub struct Blackjack {
    dealer: Vec<usize>,
    player: Vec<usize>,
    rng: StdRng,
}

impl Env for Blackjack {
    type Mask = NoMask<2>;

    fn reset(&mut self) -> (usize, Self::Mask) {
        self.dealer.clear();
        self.player.clear();
        
        self.dealer.push(BLACKJACK_CARDS[self.rng.random_range(0..13)]);
        self.dealer.push(BLACKJACK_CARDS[self.rng.random_range(0..13)]);

        self.player.push(BLACKJACK_CARDS[self.rng.random_range(0..13)]);
        self.player.push(BLACKJACK_CARDS[self.rng.random_range(0..13)]);

        let obs = Blackjack::obs_to_usize(self.get_obs());
        (obs, NoMask)
    }

    fn step(&mut self, action: usize) -> (usize, f32, bool, bool, Self::Mask) {
        let mut terminated = false;
        let mut reward = 0f32;
        
        if action == 1 {
            self.player.push(BLACKJACK_CARDS[self.rng.random_range(0..13)]);
            if Blackjack::is_bust(&self.player) {
                terminated = true;
                reward = -1f32;
            }
        } else {
            terminated = true;
            while Blackjack::sum_hand(&self.dealer) < 17 { self.dealer.push(BLACKJACK_CARDS[self.rng.random_range(0..13)]); }
            let dealer_score = Blackjack::score(&self.dealer);
            let player_score = Blackjack::score(&self.player);
            reward = if player_score > dealer_score {
                1f32
            } else if player_score == dealer_score {
                0f32
            } else {
                -1f32
            };

            if Blackjack::is_natural(&self.player) && !Blackjack::is_natural(&self.dealer) { reward = 1.5f32; }
        }

        (Blackjack::obs_to_usize(self.get_obs()), reward, terminated, false, NoMask)
    }
}

impl Blackjack {
    /// create a new Blackjack environment. `reset()` must be called before using.
    pub fn new(seed: u64) -> Self {
        Self {
            dealer: vec![],
            player: vec![],
            rng: StdRng::seed_from_u64(seed)
        }
    }
    /// returns the number of states
    pub fn n_states(&self) -> usize { 400 }
    /// returns the number of actions
    pub fn n_actions(&self) -> usize { 2 }

    fn get_obs(&self) -> (usize, usize, bool) {
        let player_sum = Blackjack::sum_hand(&self.player);
        (player_sum, self.dealer[0], Blackjack::usable_ace(&self.player))
    }

    fn obs_to_usize((player_sum, dealer_shown, usable_ace): (usize, usize, bool)) -> usize {
        let usable_ace = if usable_ace { 1 } else { 0 };
        let dealer_shown = dealer_shown - 1;
        let player_sum = player_sum - 4;

        player_sum * 20 + dealer_shown * 2 + usable_ace
    }

    fn sum_hand(hand: &[usize]) -> usize {
        if Blackjack::usable_ace(hand) {
            return hand.iter().sum::<usize>() + 10
        } else {
            return hand.iter().sum()
        }
    }

    fn usable_ace(hand: &[usize]) -> bool {
        if !Blackjack::contains_ace(hand) {
            return false;
        } else if hand.iter().sum::<usize>() + 10 <= 21 {
            return true;
        } else {
            return false;
        }
    }

    fn contains_ace(hand: &[usize]) -> bool {
        hand.contains(&1)
    }

    fn is_bust(hand: &[usize]) -> bool {
        Self::sum_hand(hand) > 21
    }

    fn is_natural(hand: &[usize]) -> bool {
        if hand.len() != 2 { return false; } 
        (hand[0] == 1 && hand[1] == 10) || (hand[0] == 10 && hand[1] == 1)
    }

    fn score(hand: &[usize]) -> usize {
        if Blackjack::is_bust(hand) { 0 } else { Self::sum_hand(hand) }
    }
}