use crate::attempts_at_framework::v1::policy::EGreedyPolicy;
use crate::attempts_at_framework::v1::policy::Policy;
use crate::attempts_at_framework::v1::state::State;
use rand::prelude::IndexedRandom;
use std::collections::HashMap;

pub struct SarsaZero {
    action_values: HashMap<String, f64>,
    policy: EGreedyPolicy,
    default_state_action_value: f64,
    step_size_parameter: f64,
    discount_rate: f64,
}

impl SarsaZero {
    pub fn new(e: f64, step_size_parameter: f64, discount_rate: f64) -> Self {
        if step_size_parameter < 0.0 || step_size_parameter > 1.0 {
            panic!("step size parameter must be between 0.0 and 1.0"); // Dose it?
        }

        if discount_rate < 0.0 || discount_rate > 1.0 {
            panic!("discount rate must be between 0.0 and 1.0");
        }

        Self {
            action_values: HashMap::new(),
            policy: EGreedyPolicy::new(e),
            default_state_action_value: 0.0,
            step_size_parameter,
            discount_rate,
        }
    }

    pub fn get_policy(&self) -> &EGreedyPolicy {
        &self.policy
    }

    pub fn lear_for_episode_count<S: State>(
        &mut self,
        episode_count: usize,
        starting_states: Vec<S>,
    ) {
        let mut rng = rand::rng();

        (0..episode_count).for_each(|_| {
            let mut state = starting_states.choose(&mut rng).unwrap().clone();
            let mut action = self.get_action_for_state(&state);

            while !state.is_terminal() {
                let state_action_id = format!("{}_{}", state.get_id(), action);

                let current_state_action_value =
                    *self.get_state_action_value_or_default(&state_action_id);

                let (reward, next_state) = state.take_action(&action);

                let next_state_action = self.get_action_for_state(&next_state);

                let next_state_action_id = format!("{}_{}", next_state.get_id(), next_state_action);

                let new_state_action_value = current_state_action_value
                    + self.step_size_parameter
                        * (reward
                            + (self.discount_rate
                                * self.get_state_action_value_or_default(&next_state_action_id))
                            - current_state_action_value);

                self.action_values
                    .insert(state_action_id, new_state_action_value);

                let possible_actions = &state.get_actions();
                let mut best_action = possible_actions[0].clone();
                let mut best_action_value = *self.get_state_action_value_or_default(&format!(
                    "{}_{}",
                    state.get_id(),
                    best_action
                ));
                (1..possible_actions.len()).for_each(|i| {
                    let action_value = self.get_state_action_value_or_default(&format!(
                        "{}_{}",
                        state.get_id(),
                        possible_actions[i]
                    ));
                    if action_value > &best_action_value {
                        best_action = possible_actions[i].clone();
                        best_action_value = *action_value;
                    }
                });

                if best_action_value != self.default_state_action_value {
                    self.policy.set_actions_for_state(
                        state.get_id().clone(),
                        possible_actions.clone(),
                        best_action,
                    );
                }

                action = next_state_action;
                state = next_state;
            }
        })
    }

    fn get_state_action_value_or_default(&mut self, state_action_id: &str) -> &f64 {
        self.action_values
            .get(state_action_id)
            .unwrap_or(&self.default_state_action_value)
    }

    fn get_action_for_state<S: State>(&self, state: &S) -> String {
        match self.policy.select_action_for_state(&state.get_id()) {
            Ok(action) => action,
            Err(_) => {
                if state.is_terminal() {
                    String::new()
                } else {
                    state
                        .get_actions()
                        .choose(&mut rand::rng())
                        .unwrap()
                        .clone()
                }
            }
        }
    }
}
