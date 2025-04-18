use crate::attempts_at_framework::v1::policy::Policy;
use crate::attempts_at_framework::v2::state::State;
use rand::prelude::IteratorRandom;

pub fn linear_differentiable_function(values: &Vec<f64>, weights: &Vec<f64>) -> f64 {
    assert_eq!(values.len(), weights.len());
    values.iter().zip(weights).map(|(v, w)| v * w).sum()
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
        .zip(weights)
        .map(|(value, weight)| weight + (gradient * value))
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
    use crate::service::x_state_walk_environment::WalkStateFactory;
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
    fn random_walk() {
        let number_of_states = 1000;

        let state_factory = WalkStateFactory::new(number_of_states, 100, 100).unwrap();

        let starting_state = state_factory.get_starting_state();
        let learned_weights = semi_gradient_td0_single_weight(
            starting_state,
            RandomPolicy::new(),
            1.0,
            0.00002,
            100000,
        );

        let data_points = (0..number_of_states)
            .map(|i| {
                let state = state_factory.generate_state_and_reward_for_id(i as i32).1;
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
            vec![(0.0, -1.0), (number_of_states as f32, 1.0)],
            ShapeStyle::from(&RED),
        ));
        line_chart_builder.add_data(LineChartData::new(
            "state values".to_string(),
            data_points,
            ShapeStyle::from(&BLUE),
        ));

        line_chart_builder.create_chart().unwrap();
    }
}
