use crate::attempts_at_framework::v1::policy::{EGreedyPolicy, Policy};
use crate::attempts_at_framework::v1::state::State;
use rand::prelude::{IndexedRandom, ThreadRng};
use std::collections::HashMap;

pub struct QLearning {
    action_values: HashMap<String, f64>,
    policy: EGreedyPolicy,
    default_action_value: f64,
    step_size_parameter: f64,
    discount_rate: f64,
}

impl QLearning {
    pub fn new(e: f64, step_size_parameter: f64, discount_rate: f64) -> Self {
        if step_size_parameter < 0.0 || step_size_parameter > 1.0 {
            panic!("Step size parameter must be between 0.0 and 1.0")
        }
        if discount_rate < 0.0 || discount_rate > 1.0 {
            panic!("Discount rate must be between 0.0 and 1.0")
        }

        Self {
            action_values: HashMap::new(),
            policy: EGreedyPolicy::new(e),
            default_action_value: 0.0,
            step_size_parameter,
            discount_rate,
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
            let mut state = starting_states.choose(&mut rng).unwrap().clone();

            while !state.is_terminal() {
                let next_state = self.learn_for_single_state(&mut state);
                state = next_state;
            }
        });
    }

    pub fn learn_for_single_state<S: State>(&mut self, mut state: &mut S) -> S {
        let mut rng = rand::rng();

        let action = match self.policy.select_action_for_state(&state.get_id()) {
            Ok(action) => action,
            Err(_) => state.get_actions().choose(&mut rng).unwrap().clone(),
        };

        let (reward, next_state) = state.take_action(&action);

        let max_next_state_action_value = if next_state.is_terminal() {
            0.0
        } else {
            next_state
                .get_actions()
                .iter()
                .map(|a| {
                    let action_id = format!("{}_{}", next_state.get_id(), a);
                    *self
                        .action_values
                        .get(&action_id)
                        .unwrap_or(&self.default_action_value)
                })
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap()
        };
        let current_state_action_value = self
            .action_values
            .get(&format!("{}_{}", state.get_id(), action))
            .unwrap_or(&self.default_action_value);

        let new_state_action_value = current_state_action_value
            + (self.step_size_parameter
                * (reward + (self.discount_rate * max_next_state_action_value)
                    - current_state_action_value));

        self.action_values.insert(
            format!("{}_{}", state.get_id(), action),
            new_state_action_value,
        );

        if new_state_action_value != self.default_action_value {
            let best_action = state
                .get_actions()
                .iter()
                .map(|a| {
                    let action_id = format!("{}_{}", state.get_id(), a);
                    (
                        a.clone(),
                        self.action_values
                            .get(&action_id)
                            .unwrap_or(&self.default_action_value),
                    )
                })
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;

            self.policy.set_actions_for_state(
                state.get_id().clone(),
                state.get_actions().clone(),
                best_action.clone(),
            );
        }
        next_state
    }
}
