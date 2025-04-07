use crate::attempts_at_framework::v1::policy::{EGreedyPolicy, Policy};
use crate::attempts_at_framework::v1::state::State;
use rand::prelude::{IndexedRandom, ThreadRng};
use std::collections::HashMap;

pub struct NStepSarsa {
    n: usize,
    policy: EGreedyPolicy,
    default_state_value: f64,
    step_size_parameter: f64,
    discount_rate: f64,
    state_action_values: HashMap<String, f64>,
    num_of_episodes_learned_for: usize,
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
            default_state_value: -10.0,
            step_size_parameter,
            discount_rate,
            state_action_values: HashMap::new(),
            num_of_episodes_learned_for: 0,
        }
    }

    pub fn get_policy(&self) -> &EGreedyPolicy {
        &self.policy
    }

    pub fn get_num_of_episodes_learned_for(&self) -> usize {
        self.num_of_episodes_learned_for
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
        let mut terminal_time: Option<usize> = None;
        let mut current_state = starting_state;
        let mut current_action = self.pick_action_for_state_based_on_policy(&current_state);
        let mut next_state: Option<S> = None;
        let mut next_action: Option<String> = None;
        let mut states_actions_and_rewards: Vec<(S, String, f64)> = Vec::new();
        states_actions_and_rewards.push((current_state.clone(), current_action.clone(), 0.0));

        for time_step in 0..usize::MAX {
            if time_step < terminal_time.unwrap_or(usize::MAX) {
                let (reward, ns) = current_state.take_action(&current_action);
                if ns.is_terminal() {
                    terminal_time = Some(time_step + 1);
                } else {
                    next_action = Some(self.pick_action_for_state_based_on_policy(&ns));
                    states_actions_and_rewards.push((ns.clone(), next_action.clone().unwrap(), reward));
                    next_state = Some(ns);
                }
            }

            let time_step_to_update = time_step as i32 - (self.n + 1) as i32;
            if time_step_to_update >= 0 {
                let f = (time_step_to_update + 1) as usize;
                let l = (time_step_to_update as usize + self.n)
                    .min(terminal_time.unwrap_or(usize::MAX));
                let sum_of_rewards = &states_actions_and_rewards[f..l]
                    .iter()
                    .enumerate()
                    .map(|(index, (_, _, r))| {
                        let pow = index as i32 - time_step_to_update - 1;
                        let idk = self.discount_rate.powi(pow);
                        idk * r
                    })
                    .sum::<f64>();

                let state_value_at_r_plus_n = if time_step_to_update as usize + self.n
                    < terminal_time.unwrap_or(usize::MAX)
                {
                    let adjusted_discount_rate = self.discount_rate.powi(self.n as i32);
                    let (s, a, _) = &states_actions_and_rewards[time_step_to_update as usize + self.n];
                    let state_action_id = Self::get_state_action_id(&s.get_id(), &a);
                    let state_value = self
                        .state_action_values
                        .get(&state_action_id)
                        .unwrap_or(&self.default_state_value);
                    state_value * adjusted_discount_rate
                } else {
                    0.0
                };

                let total_reward = sum_of_rewards + state_value_at_r_plus_n;

                let (state_to_update, action_chosen, _) = &states_actions_and_rewards[time_step_to_update as usize];
                let state_id_to_update = state_to_update.get_id();
                let state_action_id = Self::get_state_action_id(&state_id_to_update, &action_chosen);
                let existing_value = self
                    .state_action_values
                    .get(&state_action_id)
                    .unwrap_or(&self.default_state_value);
                let new_value =
                    existing_value + (self.step_size_parameter * (total_reward - existing_value));
                self.state_action_values
                    .insert(state_action_id.clone(), new_value);

                let (_, best_action) = state_to_update.get_actions().iter().fold((f64::MIN, String::new()), |(best_reward, best_action), action| {
                    let possible_state_action_id = Self::get_state_action_id(&state_id_to_update, action);
                    let possible_state_action_value = self.state_action_values.get(&possible_state_action_id).unwrap_or(&self.default_state_value);
                    if *possible_state_action_value > best_reward {
                        (*possible_state_action_value, action.clone())
                    } else {
                        (best_reward, best_action)
                    }
                });
                self.policy.set_actions_for_state(state_id_to_update.clone(), state_to_update.get_actions().clone(), best_action);
            }

            let terminal_time_as_i32 = match terminal_time {
                None => i32::MAX,
                Some(time) => time as i32,
            };

            if time_step_to_update == terminal_time_as_i32 - 1 {
                break;
            }
            current_state = next_state.unwrap_or(current_state);
            next_state = None;
            current_action = next_action.unwrap_or(current_action);
            next_action = None;
        }
        self.num_of_episodes_learned_for += 1;
    }

    fn get_state_action_id(state_id: &str, action: &str) -> String {
        format!("{}_{}", state_id, action)
    }

    fn pick_action_for_state_based_on_policy<S: State>(&self, current_state: &S) -> String {

        match self.policy.select_action_for_state(&current_state.get_id()) {
            Ok(action) => action,
            Err(_) => {
                let actions = current_state.get_actions();
                actions
                    .choose(&mut rand::rng())
                    .unwrap()
                    .clone()
            },
        }
    }
}
