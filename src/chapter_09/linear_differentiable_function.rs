use crate::attempts_at_framework::v1::policy::Policy;
use crate::attempts_at_framework::v2::state::State;
use rand::prelude::IteratorRandom;

pub fn linear_differentiable_function(values: &Vec<f64>, weights: &Vec<f64>) -> f64 {
    assert_eq!(values.len(), weights.len());
    (0..values.len()).fold(0.0, |acc, index| acc + (values[index] * weights[index]))
}

pub fn weight_update(
    values: &Vec<f64>,
    weights: &Vec<f64>,
    learning_rate: f64,
    expected_value: f64,
) -> Vec<f64> {
    assert_eq!(values.len(), weights.len());
    let current_value = linear_differentiable_function(values, weights);
    let error = expected_value - current_value;
    let gradient = error * learning_rate;
    values
        .iter()
        .enumerate()
        .map(|(index, value)| weights[index] + (gradient * value))
        .collect()
}

pub fn semi_gradient_td0_single_weight<S: State, P: Policy>(
    starting_state: S,
    policy: P,
    discount_rate: f64,
    learning_rate: f64,
    episode_count: usize,
) -> Vec<f64> {
    let mut weights = vec![0.0; starting_state.get_values().len()];

    (0..episode_count).for_each(|_| {
        let mut current_state = starting_state.clone();
        while !current_state.is_terminal() {
            let action = match policy.select_action_for_state(&current_state.get_id()) {
                Ok(a) => a,
                Err(_) => current_state
                    .get_actions()
                    .iter()
                    .choose(&mut rand::rng())
                    .unwrap()
                    .clone(),
            };

            let (reward, next_state) = current_state.take_action(&action);

            // println!("current: {}, action: {}, reward: {}, next: {}", current_state.get_id(), action, reward, next_state.get_id());

            let new_state_value =
                linear_differentiable_function(&next_state.get_values(), &weights);
            let expected = reward + (discount_rate * new_state_value);

            let new_weights = weight_update(
                &current_state.get_values(),
                &weights,
                learning_rate,
                expected,
            );
            weights = new_weights;

            current_state = next_state;
        }
    });

    weights
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attempts_at_framework::v1::policy::RandomPolicy;
    use crate::service::{LineChartBuilder, LineChartData};
    use plotters::prelude::{ShapeStyle, RED};
    use plotters::style::BLUE;
    use std::path::PathBuf;

    #[test]
    fn test_simple_case() {
        let values = vec![1.0, 2.0, 3.0];
        let weights = vec![1.0, 2.0, 3.0];
        let result = linear_differentiable_function(&values, &weights);
        assert_eq!(result, 14.0);
    }

    #[test]
    fn weight_update_test() {
        let learning_rate = 0.1;
        let values = vec![1.0, 2.0];
        let mut weights = vec![1.0, 1.0];
        let expected_state_value = 10.0;

        let mut current_value = linear_differentiable_function(&values, &weights);
        let mut count = 0;

        while (current_value - expected_state_value).abs() > 0.00001 {
            if count > 1000 {
                panic!("Failed to converge");
            }
            weights = weight_update(&values, &weights, learning_rate, expected_state_value);
            current_value = linear_differentiable_function(&values, &weights);
            count += 1;
        }

        println!(
            "weights: {:?}, converged after: {} iterations",
            weights, count
        );
    }

    #[test]
    fn graph_learning_rates() {
        let learning_rates = vec![0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0];

        let initial_weights = vec![1.0, 1.0];
        let values = vec![1.0, 2.0];
        let expected_state_value = -10.0;

        let mut chart_builder = LineChartBuilder::new();
        chart_builder
            .set_path(PathBuf::from(
                "output/chapter9/learning_rates_and_linear_regression_convergence.png",
            ))
            .set_title("Learning rates vs iterations to convergence".to_string())
            .set_x_label("Learning rate".to_string())
            .set_y_label("Iterations to convergence".to_string());

        let mut data: Vec<(f32, f32)> = Vec::new();

        for learning_rate in learning_rates {
            let mut weights = initial_weights.clone();
            let mut current_value = linear_differentiable_function(&values, &weights);
            let mut count = 0;

            while (current_value - expected_state_value).abs() > 0.00001 {
                if count > 1000 {
                    break;
                }
                weights = weight_update(&values, &weights, learning_rate, expected_state_value);
                current_value = linear_differentiable_function(&values, &weights);
                count += 1;
            }

            println!(
                "learning rate: {}, weights: {:?}, converged after: {} iterations",
                learning_rate, weights, count
            );

            data.push((learning_rate as f32, count as f32));
        }

        let data_as_line = LineChartData::new("idk".to_string(), data, ShapeStyle::from(&BLUE));
        chart_builder.add_data(data_as_line);

        chart_builder.create_chart().unwrap()
    }

    #[test]
    fn thousand_step_walk() {
        let starting_state = HundredStepState::new(500);

        let learned_weights = semi_gradient_td0_single_weight(
            starting_state,
            RandomPolicy::new(),
            1.0,
            0.00002,
            1000000,
        );

        let data_points = (0..1000)
            .map(|i| {
                let state = HundredStepState::new(i);
                let value = linear_differentiable_function(&state.get_values(), &learned_weights);
                (i as f32, value as f32)
            })
            .collect();

        println!("learned weights: {:?}", learned_weights);

        let mut line_chart_builder = LineChartBuilder::new();
        line_chart_builder.set_path(PathBuf::from(
            "output/chapter9/thousand_state_random_walk.png",
        ));
        line_chart_builder.set_title("thousand state random walk".to_string());
        line_chart_builder.set_x_label("State".to_string());
        line_chart_builder.set_y_label("Value".to_string());
        line_chart_builder.add_data(LineChartData::new(
            "Expected".to_string(),
            vec![(0.0, -1.0), (1000.0, 1.0)],
            ShapeStyle::from(&RED),
        ));
        line_chart_builder.add_data(LineChartData::new(
            "state values".to_string(),
            data_points,
            ShapeStyle::from(&BLUE),
        ));

        line_chart_builder.create_chart().unwrap();
    }

    #[derive(Clone, Debug)]
    struct HundredStepState {
        id: i32,
    }

    impl HundredStepState {
        fn new(id: i32) -> Self {
            Self { id }
        }
    }

    impl State for HundredStepState {
        fn get_id(&self) -> String {
            self.id.to_string()
        }

        fn get_actions(&self) -> Vec<String> {
            (-100..100).map(|i| i.to_string()).collect()
        }

        fn is_terminal(&self) -> bool {
            if self.id < 0 || self.id > 1000 {
                true
            } else {
                false
            }
        }

        fn take_action(&self, action: &str) -> (f64, Self) {
            let diff: i32 = action.parse().unwrap();
            let new_id = self.id + diff;
            let new_state = HundredStepState::new(new_id);
            let reward = if new_id < 0 {
                -1.0
            } else if new_id > 1000 {
                1.0
            } else {
                0.0
            };
            (reward, new_state)
        }

        fn get_values(&self) -> Vec<f64> {
           let index = (self.id / 100).clamp(0, 9) as usize;
            let mut res_vec = vec![0.0; 10];
            res_vec[index] = 1.0;
            res_vec
        }
    }
}
