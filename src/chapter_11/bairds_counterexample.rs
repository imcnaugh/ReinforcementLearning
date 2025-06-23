use crate::attempts_at_framework::v1::policy::{DeterministicPolicy, StochasticPolicy};
use crate::attempts_at_framework::v2::state::State;
use rand::prelude::IteratorRandom;

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
        vec!["dashed".to_string(), "solid".to_string()]
    }

    fn is_terminal(&self) -> bool {
        false
    }

    fn take_action(&self, action: &str) -> (f64, Self) {
        let actions = self.get_actions();
        if !actions.contains(&action.to_string()) {
            panic!("Invalid action");
        }
        if action == "solid" {
            (0.0, Self::new(7))
        } else {
            let next_id = (1..7).choose(&mut rand::rng()).unwrap();
            (0.0, Self::new(next_id))
        }
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
        target_policy.set_actions_for_state(id.to_string(), "solid".to_string());
    });
    target_policy
}

fn create_behavior_policy() -> StochasticPolicy {
    let mut behavior_policy = StochasticPolicy::new();
    (1..=7).for_each(|id| {
        behavior_policy.set_actions_for_state(
            id.to_string(),
            vec![
                ("dashed".to_string(), 6.0 / 7.0),
                ("solid".to_string(), 1.0 / 7.0),
            ],
        );
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
        .map(|(w, dw)| w + (c * dw))
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
    use crate::service::{LineChartBuilder, LineChartData};
    use rand::prelude::IteratorRandom;
    use std::path::PathBuf;

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
            let expected_actions = vec!["dashed", "solid"];
            assert_eq!(actions, expected_actions);
        });
    }

    #[test]
    fn test_take_action() {
        (2..=7).for_each(|id| {
            let state = TestState::new(id);
            let (reward, next_state) = state.take_action("solid");
            assert_eq!(reward, 0.0);
            assert_eq!(next_state.id, 7);
        });
    }

    #[test]
    fn test_create_target_policy() {
        let policy = create_target_policy();
        (1..=7).for_each(|id| {
            let id = id.to_string();
            assert_eq!("solid", policy.select_action_for_state(&id).unwrap());
        })
    }

    #[test]
    fn test_create_behavior_policy() {
        let policy = create_behavior_policy();
        let choice = policy.select_action_for_state("1").unwrap();
        let possible = vec!["solid", "dashed"];
        assert!(possible.contains(&choice.as_str()))
    }

    #[test]
    fn test_td_error() {
        let weights = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 1.0];
        let discount_factor = 1.0;
        let current_state = TestState::new(1);
        let next_state = TestState::new(7);
        let reward = 0.0;
        let td_error = get_td_error(
            &weights,
            reward,
            discount_factor,
            &current_state,
            &next_state,
        );
        assert_eq!(td_error, 9.0);
    }

    #[test]
    fn test_update_weights() {
        let weights = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 1.0];
        let learning_rate = 0.1;
        let importance_sampling_ratio = 1.0;
        let td_error = 9.0;
        let partial_derivatives = TestState::new(1).get_values();
        let updated_weights = update_weights(
            &weights,
            learning_rate,
            importance_sampling_ratio,
            td_error,
            &partial_derivatives,
        );
        assert_eq!(
            updated_weights,
            vec![2.8, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 1.9]
        );
    }

    #[test]
    fn test_updated_weights_closer_to_next_state() {
        let weights = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 1.0];
        let learning_rate = 0.1;
        let discount_factor = 1.0;
        let current_state = TestState::new(1);
        let next_state = TestState::new(7);
        let reward = 0.0;
        let td_error = get_td_error(
            &weights,
            reward,
            discount_factor,
            &current_state,
            &next_state,
        );
        let partial_derivatives = current_state.get_values();
        let updated_weights = update_weights(
            &weights,
            learning_rate,
            learning_rate,
            td_error,
            &partial_derivatives,
        );

        let updated_td_error = get_td_error(
            &updated_weights,
            reward,
            discount_factor,
            &current_state,
            &next_state,
        );

        assert!(updated_td_error < td_error);
    }

    #[test]
    fn test_td_0_off_policy_instability() {
        let mut weights = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 1.0];
        let size_step = 0.01;
        let discount_factor = 0.99;

        let total_steps = 1000;

        let starting_state = (0..=7).choose(&mut rand::rng()).unwrap();
        let mut state = TestState::new(starting_state);

        let behavior_policy = create_behavior_policy();
        let _target_policy = create_target_policy();

        let mut weights_over_time = vec![vec![]; 8];

        (0..total_steps).for_each(|_| {
            let next_action = behavior_policy
                .select_action_for_state(&state.get_id())
                .unwrap();
            let (reward, next_state) = state.take_action(&next_action);

            let importance_sampling_ratio = if next_action == "solid" { 7.0 } else { 0.0 };

            let error = get_td_error(&weights, reward, discount_factor, &state, &next_state);

            weights = update_weights(
                &weights,
                size_step,
                importance_sampling_ratio,
                error,
                &state.get_values(),
            );
            state = next_state;
            weights.iter().enumerate().for_each(|(i, w)| {
                weights_over_time[i].push(*w);
            })
        });

        let mut chart_builder = LineChartBuilder::new();
        chart_builder
            .set_path(PathBuf::from(
                "output/chapter11/baird's counter example weights.png",
            ))
            .set_x_label("steps".to_string())
            .set_y_label("weight value".to_string());

        weights_over_time.iter().enumerate().for_each(|(i, w)| {
            let data: Vec<(f32, f32)> = w
                .iter()
                .enumerate()
                .map(|(i, w)| (i as f32, w.clone() as f32))
                .collect();
            let data = LineChartData::new(format!("weight {}", i + 1), data);
            chart_builder.add_data(data);
        });

        chart_builder.create_chart().unwrap()
    }

    #[test]
    fn test_on_policy_stability() {
        let mut weights = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 1.0];
        let size_step = 0.01;
        let discount_factor = 0.99;

        let total_steps = 1000;

        let starting_state = (0..=7).choose(&mut rand::rng()).unwrap();
        let mut state = TestState::new(starting_state);

        let behavior_policy = create_behavior_policy();

        let mut estimated_state_values = vec![vec![]; 7];

        (0..total_steps).for_each(|_| {
            let next_action = behavior_policy
                .select_action_for_state(&state.get_id())
                .unwrap();
            let (reward, next_state) = state.take_action(&next_action);

            let error = get_td_error(&weights, reward, discount_factor, &state, &next_state);

            weights = update_weights(&weights, size_step, 1.0, error, &state.get_values());
            state = next_state;

            (0..7).for_each(|i| {
                let s = TestState::new(i);
                let value = s.get_values();
                let estimated_value: f64 = weights.iter().zip(value).map(|(w, v)| w * v).sum();
                estimated_state_values[i].push(estimated_value);
            })
        });

        (0..7).for_each(|i| {
            let s = TestState::new(i);
            let value = s.get_values();
            let estimated_value: f64 = weights.iter().zip(value).map(|(w, v)| w * v).sum();
            println!(
                "final estimated state value for state: {}, {}",
                i + 1,
                estimated_value
            )
        });

        let mut chart_builder = LineChartBuilder::new();
        chart_builder
            .set_path(PathBuf::from(
                "output/chapter11/baird's counter example on policy weights.png",
            ))
            .set_x_label("steps".to_string())
            .set_y_label("estimated state value".to_string());

        estimated_state_values
            .iter()
            .enumerate()
            .for_each(|(i, w)| {
                let data: Vec<(f32, f32)> = w
                    .iter()
                    .enumerate()
                    .map(|(i, w)| (i as f32, w.clone() as f32))
                    .collect();
                let data = LineChartData::new(format!("state {}", i + 1), data);
                chart_builder.add_data(data);
            });

        chart_builder.create_chart().unwrap()
    }
}
