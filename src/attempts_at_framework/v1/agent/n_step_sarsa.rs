use crate::attempts_at_framework::v1::policy::{EGreedyPolicy, Policy};
use crate::attempts_at_framework::v1::state::State;
use rand::prelude::IndexedRandom;
use std::collections::HashMap;

pub struct NStepSarsa {
    n: usize,
    policy: EGreedyPolicy,
    default_state_value: f64,
    step_size_parameter: f64,
    discount_rate: f64,
    state_values: HashMap<String, f64>,
}

impl NStepSarsa {
    pub fn new(n: usize, e: f64, step_size_parameter: f64, discount_rate: f64) -> Self {
        if step_size_parameter < 0.0 || step_size_parameter > 1.0 {
            panic!("Step size parameter must be between 0.0 and 1.0")
        }
        if discount_rate < 0.0 || discount_rate > 1.0 {
            panic!("Discount rate must be between 0.0 and 1.0")
        }

        Self {
            n,
            policy: EGreedyPolicy::new(e),
            default_state_value: 0.0,
            step_size_parameter,
            discount_rate,
            state_values: HashMap::new(),
        }
    }

    pub fn get_policy(&self) -> &EGreedyPolicy {
        &self.policy
    }

    pub fn learn_for_episode_count<S: State>(
        &mut self,
        episode_count: usize,
        starting_states: Vec<S>,
    ) {
        let mut rng = rand::rng();

        (0..episode_count).for_each(|_| {
            let starting_state = starting_states.choose(&mut rng).unwrap().clone();
            self.learn_from_episode(starting_state);
        });
    }

    pub fn learn_from_episode<S: State>(&mut self, starting_state: S) {
        let mut rng = rand::rng();
        let mut terminal_time: Option<usize> = None;
        let mut current_state = starting_state;
        let mut next_state: Option<S> = None;
        let mut states_and_rewards: Vec<(S, f64)> = Vec::new();
        states_and_rewards.push((current_state.clone(), 0.0));

        for time_step in 0..usize::MAX {
            if time_step < terminal_time.unwrap_or(usize::MAX) {
                let current_state_id = current_state.get_id();
                let action = match self.policy.select_action_for_state(&current_state_id) {
                    Ok(action) => action,
                    Err(_) => current_state
                        .get_actions()
                        .choose(&mut rng)
                        .unwrap()
                        .clone(),
                };
                let (reward, ns) = current_state.take_action(&action);
                states_and_rewards.push((ns.clone(), reward));
                if ns.is_terminal() {
                    terminal_time = Some(time_step + 1);
                }
                next_state = Some(ns);
            }

            let time_step_to_update = time_step as i32 - (self.n + 1) as i32;
            if time_step_to_update >= 0 {
                let f = (time_step_to_update + 1) as usize;
                let l = (time_step_to_update as usize + self.n)
                    .min(terminal_time.unwrap_or(usize::MAX));
                let sum_of_rewards = &states_and_rewards[f..l]
                    .iter()
                    .enumerate()
                    .map(|(index, (_, r))| {
                        let pow = index as i32 - time_step_to_update - 1;
                        let idk = self.discount_rate.powi(pow);
                        idk * r
                    })
                    .sum::<f64>();

                let state_value_at_r_plus_n = if time_step_to_update as usize + self.n
                    < terminal_time.unwrap_or(usize::MAX)
                {
                    let adjusted_discount_rate = self.discount_rate.powi(self.n as i32);
                    let (s, _) = &states_and_rewards[time_step_to_update as usize + self.n];
                    let state_value = self
                        .state_values
                        .get(&s.get_id())
                        .unwrap_or(&self.default_state_value);
                    state_value * adjusted_discount_rate
                } else {
                    0.0
                };

                let total_reward = sum_of_rewards + state_value_at_r_plus_n;

                let state_to_update = &states_and_rewards[time_step_to_update as usize].0;
                let state_id_to_update = state_to_update.get_id();
                let existing_value = self
                    .state_values
                    .get(&state_id_to_update)
                    .unwrap_or(&self.default_state_value);
                let new_value =
                    existing_value + (self.step_size_parameter * (total_reward - existing_value));
                self.state_values
                    .insert(state_id_to_update.clone(), new_value);

                // let best_action = state_to_update.get_actions().iter().fold((f64::MIN, String::new()), |(best_reward, best_action), action| {
                //
                // });
            }

            if time_step_to_update as usize == terminal_time.unwrap_or(usize::MAX) - 1 {
                break;
            }
            current_state = next_state.unwrap_or(current_state);
            next_state = None;
        }
    }
}
