use crate::attempts_at_framework::v1::policy::{DeterministicPolicy, StochasticPolicy};
use crate::attempts_at_framework::v2::state::State;
use rand::seq::WeightError;

#[derive(Clone)]
struct TestState {
    id: usize,
}

impl TestState {
    fn new(id: usize) -> Self {
        let id = id.clamp(1, 7);
        Self { id }
    }
}

impl State for TestState {
    fn get_id(&self) -> String {
        self.id.to_string()
    }

    fn get_actions(&self) -> Vec<String> {
        let mut options = vec!["1", "2", "3", "4", "5", "6", "7"];
        let index_to_remove = self.id - 1;
        options.remove(index_to_remove);
        options.iter().map(|s| s.to_string()).collect()
    }

    fn is_terminal(&self) -> bool {
        false
    }

    fn take_action(&self, action: &str) -> (f64, Self) {
        if !self.get_actions().contains(&action.to_string()) {
            panic!("Invalid action");
        }
        let action = action.parse::<usize>().unwrap();
        let next_state = TestState::new(action);
        (0.0, next_state)
    }

    fn get_values(&self) -> Vec<f64> {
        let mut values = vec![0.0; 8];
        if self.id == 7 {
            values[6] = 1.0;
            values[7] = 2.0;
            values
        } else {
            let index = self.id - 1;
            values[index] = 2.0;
            values[7] = 1.0;
            values
        }
    }
}

fn create_target_policy() -> DeterministicPolicy {
    let mut target_policy = DeterministicPolicy::new();
    (1..=7).for_each(|id| {
        target_policy.set_actions_for_state(id.to_string(), "7".to_string());
    });
    target_policy
}

fn create_behavior_policy() -> StochasticPolicy {
    let mut behavior_policy = StochasticPolicy::new();
    (1..=7).for_each(|id| {
        let possibilities = 1.0 / 7.0;
        let possible_actions = vec!["1", "2", "3", "4", "5", "6", "7"]
            .iter()
            .map(|action| (action.to_string(), possibilities))
            .collect();
        behavior_policy.set_actions_for_state(id.to_string(), possible_actions);
    });
    behavior_policy
}

fn update_weights(
    weights: &Vec<f64>,
    learning_rate: f64,
    importance_sampling_ratio: f64,
    td_error: f64,
    partial_derivatives: &Vec<f64>,
) -> Vec<f64> {
    let c = learning_rate * importance_sampling_ratio * td_error;
    weights
        .iter()
        .zip(partial_derivatives)
        .map(|(w, dw)| w + c * dw)
        .collect()
}

fn get_td_error(
    weights: &Vec<f64>,
    reward: f64,
    discount_factor: f64,
    current_state: &TestState,
    next_state: &TestState,
) -> f64 {
    let get_value_estimate = |state: &TestState, weights: &Vec<f64>| -> f64 {
        state
            .get_values()
            .iter()
            .zip(weights)
            .map(|(v, w)| v * w)
            .sum()
    };

    reward + (discount_factor * get_value_estimate(next_state, weights))
        - (get_value_estimate(current_state, weights))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attempts_at_framework::v1::policy::Policy;
    use rand::prelude::IteratorRandom;

    #[test]
    fn test_get_values() {
        (1..7).for_each(|id| {
            let state = TestState::new(id);
            let values = state.get_values();
            let mut expected_values = vec![0.0; 8];
            expected_values[id - 1] = 2.0;
            expected_values[7] = 1.0;
            assert_eq!(values, expected_values);
        });

        let state = TestState::new(7);
        let values = state.get_values();
        let mut expected_values = vec![0.0; 8];
        expected_values[6] = 1.0;
        expected_values[7] = 2.0;
        assert_eq!(values, expected_values);
    }

    #[test]
    fn test_get_actions() {
        (1..=7).for_each(|id| {
            let state = TestState::new(id);
            let actions = state.get_actions();
            let mut expected_actions = vec!["1", "2", "3", "4", "5", "6", "7"];
            let index_to_remove = id - 1;
            expected_actions.remove(index_to_remove);
            assert_eq!(actions, expected_actions);
        });
    }

    #[test]
    fn test_take_action() {
        (2..=7).for_each(|id| {
            let state = TestState::new(id);
            let (reward, next_state) = state.take_action("1");
            assert_eq!(reward, 0.0);
            assert_eq!(next_state.id, 1);
        });
    }

    #[test]
    fn test_create_target_policy() {
        let policy = create_target_policy();
        (1..=7).for_each(|id| {
            let id = id.to_string();
            assert_eq!("7", policy.select_action_for_state(&id).unwrap());
        })
    }

    #[test]
    fn test_create_behavior_policy() {
        let policy = create_behavior_policy();
        let choice = policy.select_action_for_state("1").unwrap();
        let possible = vec!["1", "2", "3", "4", "5", "6", "7"];
        assert!(possible.contains(&choice.as_str()))
    }

    #[test]
    fn test_td_0_off_policy_instability() {
        let mut weights = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 1.0];
        let size_step = 0.01;

        let total_steps = 1000;

        let starting_state = (0..=7).choose(&mut rand::rng()).unwrap();
        let mut state = TestState::new(starting_state);
    }
}
