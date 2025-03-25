use crate::chapter_05::policy::Policy;
use crate::chapter_05::race_track::state::State;
use rand::prelude::*;
use rand::Rng;
use std::collections::HashMap;

pub struct MonteCarloLearner<'a, S: State, P: Policy> {
    state_action_values: HashMap<String, f32>,
    state_action_cumulative_rewards: HashMap<String, f32>,
    starting_states: Vec<S>,
    discount_rate: f64,
    target_policy: &'a P,
    behavior_policy: &'a P,
}

impl<'a, S: State, P: Policy> MonteCarloLearner<'a, S, P> {
    pub fn new(
        starting_states: Vec<S>,
        discount_rate: f64,
        target_policy: &'a P,
        behavior_policy: &'a P,
    ) -> Self {
        Self {
            state_action_values: HashMap::new(),
            state_action_cumulative_rewards: HashMap::new(),
            starting_states,
            discount_rate,
            target_policy,
            behavior_policy,
        }
    }

    pub fn get_target_policy(&self) -> &P {
        &self.target_policy
    }

    pub fn learn_for_episodes(&self, episode_count: usize) {
        (0..episode_count).for_each(|_| {})
    }

    fn generate_episode(&self) {
        let mut rng = rand::rng();
        let starting_state = self.starting_states.choose(&mut rng).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_state_selection() {
        let mut rng = rand::rng();
        let vec: Vec<i32> = (0..100).collect();
        let random_state = vec.choose(&mut rng).unwrap();
        println!("{}", random_state);
    }
}
