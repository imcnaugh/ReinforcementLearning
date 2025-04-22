use crate::attempts_at_framework::v1::policy::Policy;
use crate::attempts_at_framework::v2::state::State;
use egui::Key::N;
use rand::prelude::IteratorRandom;
use std::collections::VecDeque;

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

pub fn monte_carlo_stochastic_gradient_decent<S: State, P: Policy>(
    starting_state: S,
    policy: P,
    learning_rate: f64,
    episode_count: usize,
) -> Vec<f64> {
    let mut weights = vec![0.0; starting_state.get_values().len()];

    (0..episode_count).for_each(|_| {
        let mut current_state = starting_state.clone();
        let mut states_and_rewards: Vec<(S, f64)> = Vec::new();

        while !current_state.is_terminal() {
            let action = select_action(&current_state, &policy);
            let (reward, next_state) = current_state.take_action(&action);
            states_and_rewards.push((next_state.clone(), reward));
            current_state = next_state;
        }

        let mut total_reward = states_and_rewards.iter().map(|(_, r)| r).sum::<f64>();
        states_and_rewards.iter().for_each(|(state, reward)| {
            let new_weights =
                weight_update(&state.get_values(), &weights, learning_rate, total_reward);
            weights = new_weights;
            total_reward = total_reward - reward;
        });
    });
    weights
}

pub fn semi_gradient_td0<S: State, P: Policy>(
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
            let action = select_action(&starting_state, &policy);

            let (reward, next_state) = current_state.take_action(&action);

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

pub fn n_step_semi_gradient_td<S: State, P: Policy>(
    starting_state: S,
    policy: P,
    discount_rate: f64,
    learning_rate: f64,
    episode_count: usize,
    n: usize,
) -> Vec<f64> {
    let mut weights = vec![0.0; starting_state.get_values().len()];

    (0..episode_count).for_each(|_| {
        let mut termination_time: Option<i32> = None;
        let mut rewards: Vec<f64> = vec![0.0];
        let mut states: Vec<S> = vec![starting_state.clone()];

        let mut current_state = starting_state.clone();

        for current_timestep in 0..i32::MAX {
            if current_timestep < termination_time.unwrap_or(i32::MAX) {
                let action = select_action(&starting_state, &policy);
                let (reward, next_state) = current_state.take_action(&action);
                rewards.push(reward);
                states.push(next_state.clone());

                if next_state.is_terminal() {
                    termination_time = Some(current_timestep + 1);
                }

                current_state = next_state;
            }
            let time_step_to_update = current_timestep - n as i32 + 1;
            if time_step_to_update >= 0 {
                let start_time = time_step_to_update as usize + 1;
                let end_time = (time_step_to_update + n as i32)
                    .min(termination_time.unwrap_or(i32::MAX))
                    as usize;
                let mut expected_value = rewards[start_time..=end_time]
                    .iter()
                    .enumerate()
                    .map(|(index, r)| discount_rate.powi(index as i32) * r)
                    .sum::<f64>();

                if (time_step_to_update + n as i32) < termination_time.unwrap_or(i32::MAX) {
                    expected_value += learning_rate.powi(n as i32)
                        * linear_differentiable_function(&current_state.get_values(), &weights);
                }
                let new_weights = weight_update(
                    &states[time_step_to_update as usize].get_values(),
                    &weights,
                    learning_rate,
                    expected_value,
                );
                weights = new_weights;
            }

            if time_step_to_update == (termination_time.unwrap_or(i32::MAX) - 1) {
                break;
            }
        }
    });

    weights
}

fn select_action<S: State, P: Policy>(starting_state: &S, policy: &P) -> String {
    match policy.select_action_for_state(&starting_state.get_id()) {
        Ok(a) => a,
        Err(_) => starting_state
            .get_actions()
            .iter()
            .choose(&mut rand::rng())
            .unwrap()
            .clone(),
    }
}

pub fn n_step_semi_gradient_td_my_refactor<S: State, P: Policy>(
    starting_state: S,
    policy: P,
    discount_rate: f64,
    learning_rate: f64,
    episode_count: usize,
    n: usize,
) -> Vec<f64> {
    let mut weights = vec![0.0; starting_state.get_values().len()];

    (0..episode_count).for_each(|_| {
        let mut current_state = starting_state.clone();
        let mut next_state: Option<S> = None;
        let mut queue: VecDeque<Option<(f64, Vec<f64>)>> = VecDeque::new();
        (0..n).for_each(|_| queue.push_front(None));
        let mut sliding_reward_total = 0.0;

        while !queue.is_empty() {
            if !current_state.is_terminal() {
                let action = select_action(&current_state, &policy);
                let (reward, ns) = current_state.take_action(&action);
                if !ns.is_terminal() {
                    queue.push_front(Some((reward, current_state.get_values().clone())));
                }
                next_state = Some(ns);
                sliding_reward_total += reward;
            }
            let back_of_queue = queue.pop_back().unwrap();
            if let Some((reward, values)) = back_of_queue {
                let expected = sliding_reward_total
                    + (discount_rate.powi(n as i32)
                        * linear_differentiable_function(&current_state.get_values(), &weights));
                let new_weights = weight_update(&values, &weights, learning_rate, expected);
                weights = new_weights;

                let reward_to_remove_from_sliding_window = discount_rate.powi(n as i32) * reward;
                sliding_reward_total -= reward_to_remove_from_sliding_window;
            }

            if let Some(ns) = next_state {
                current_state = ns;
                next_state = None;
            }
            sliding_reward_total *= discount_rate;
        }
    });

    weights
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attempts_at_framework::v1::policy::RandomPolicy;
    use crate::service::x_state_walk_environment::{WalkState, WalkStateFactory};
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
    fn random_walk_monte_carlo() {
        let number_of_states = 1000;

        // let value_function = generate_state_aggregation_value_function(number_of_states, 100);
        let value_function = generate_polynomial_value_function(number_of_states, 1);
        let state_factory = WalkStateFactory::new(number_of_states, 100, &value_function).unwrap();

        let starting_state = state_factory.get_starting_state();
        let learned_weights = monte_carlo_stochastic_gradient_decent(
            starting_state,
            RandomPolicy::new(),
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
            "output/chapter9/thousand_state_random_walk-monte_carlo.png",
        ));
        line_chart_builder.set_title("Thousand state random walk Monte Carlo".to_string());
        line_chart_builder.set_x_label("State".to_string());
        line_chart_builder.set_y_label("Value".to_string());
        line_chart_builder.add_data(LineChartData::new(
            "Expected".to_string(),
            vec![(0.0, -1.0), (number_of_states as f32, 1.0)],
            ShapeStyle::from(&RED),
        ));
        line_chart_builder.add_data(LineChartData::new(
            "State values".to_string(),
            data_points,
            ShapeStyle::from(&BLUE),
        ));

        line_chart_builder.create_chart().unwrap();
    }

    #[test]
    fn random_walk_semi_gradient_td0() {
        let number_of_states = 1000;
        let discount_rate = 1.0;
        let learning_rate = 0.002;
        let episode_count = 10000;
        // let value_function = generate_simple_value_function(number_of_states);
        // let value_function = generate_state_aggregation_value_function(number_of_states, 100);
        let value_function = generate_polynomial_value_function(number_of_states, 1);

        let state_factory = WalkStateFactory::new(number_of_states, 100, &value_function).unwrap();

        let starting_state = state_factory.get_starting_state();
        let learned_weights = semi_gradient_td0(
            starting_state,
            RandomPolicy::new(),
            discount_rate,
            learning_rate,
            episode_count,
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
            "output/chapter9/thousand_state_random_walk-td_0.png",
        ));
        line_chart_builder.set_title("Thousand state random walk TD0".to_string());
        line_chart_builder.set_x_label("State".to_string());
        line_chart_builder.set_y_label("Value".to_string());
        line_chart_builder.add_data(LineChartData::new(
            "Expected".to_string(),
            vec![(0.0, -1.0), (number_of_states as f32, 1.0)],
            ShapeStyle::from(&RED),
        ));
        line_chart_builder.add_data(LineChartData::new(
            "State values".to_string(),
            data_points,
            ShapeStyle::from(&BLUE),
        ));

        line_chart_builder.create_chart().unwrap();
    }

    #[test]
    fn random_walk_semi_gradient_n_step() {
        let number_of_states = 1000;
        let discount_rate = 1.0;
        let learning_rate = 0.001;
        let episode_count = 10000;
        let n = 10;

        // let value_function = generate_simple_value_function(number_of_states);
        let value_function = generate_state_aggregation_value_function(number_of_states, 50);

        let state_factory = WalkStateFactory::new(number_of_states, 100, &value_function).unwrap();

        let starting_state = state_factory.get_starting_state();
        let learned_weights = n_step_semi_gradient_td(
            starting_state,
            RandomPolicy::new(),
            discount_rate,
            learning_rate,
            episode_count,
            n,
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
            "output/chapter9/thousand_state_random_walk-n_step.png",
        ));
        line_chart_builder.set_title(format!("Thousand state random walk {} step", n));
        line_chart_builder.set_x_label("State".to_string());
        line_chart_builder.set_y_label("Value".to_string());
        line_chart_builder.add_data(LineChartData::new(
            "Expected".to_string(),
            vec![(0.0, -1.0), (number_of_states as f32, 1.0)],
            ShapeStyle::from(&RED),
        ));
        line_chart_builder.add_data(LineChartData::new(
            "State values".to_string(),
            data_points,
            ShapeStyle::from(&BLUE),
        ));

        line_chart_builder.create_chart().unwrap();
    }

    #[test]
    fn random_walk_semi_gradient_n_step_from_refactor() {
        let number_of_states = 1000;
        let discount_rate = 1.0;
        let learning_rate = 0.0002;
        let episode_count = 10000;
        let n = 10;

        // let value_function = generate_simple_value_function(number_of_states);
        let value_function = generate_polynomial_value_function(number_of_states, 1);

        let state_factory = WalkStateFactory::new(number_of_states, 100, &value_function).unwrap();

        let starting_state = state_factory.get_starting_state();
        let learned_weights = n_step_semi_gradient_td_my_refactor(
            starting_state,
            RandomPolicy::new(),
            discount_rate,
            learning_rate,
            episode_count,
            n,
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
            "output/chapter9/thousand_state_random_walk-n_step_refactored.png",
        ));
        line_chart_builder.set_title(format!(
            "Thousand state random walk {} step - Refactored",
            n
        ));
        line_chart_builder.set_x_label("State".to_string());
        line_chart_builder.set_y_label("Value".to_string());
        line_chart_builder.add_data(LineChartData::new(
            "Expected".to_string(),
            vec![(0.0, -1.0), (number_of_states as f32, 1.0)],
            ShapeStyle::from(&RED),
        ));
        line_chart_builder.add_data(LineChartData::new(
            "State values".to_string(),
            data_points,
            ShapeStyle::from(&BLUE),
        ));

        line_chart_builder.create_chart().unwrap();
    }

    fn generate_state_aggregation_value_function(
        total_states: usize,
        group_size: usize,
    ) -> impl Fn(WalkState) -> Vec<f64> {
        let number_of_groups = total_states / group_size;

        move |state| {
            if state.is_terminal() {
                return vec![0.0; number_of_groups];
            }

            let group_id = state.get_id().parse::<usize>().unwrap() / group_size;
            let mut response = vec![0.0; number_of_groups];
            response[group_id] = 1.0;
            response
        }
    }

    fn generate_polynomial_value_function(
        total_states: usize,
        polynomial_degree: usize,
    ) -> impl Fn(WalkState) -> Vec<f64> {
        move |state| {
            let mut response = vec![0.0; polynomial_degree + 1];
            if !state.is_terminal() {
                response[0] = 1.0;
                for i in 0..polynomial_degree {
                    response[i + 1] = (state.get_id().parse::<f64>().unwrap()
                        / total_states as f64)
                        * response[i];
                }
            }
            response
        }
    }

    /// I can't get this to work, I would expect the learned weights to become -1 and .002
    /// but for whatever reason I always wind up with NaN's. I'm assuming this is due to
    /// errors with floating point arithmetic. But it's been hard to prove.
    #[deprecated = "This doesn't work, I'm not sure why"]
    fn generate_simple_value_function(total_states: usize) -> impl Fn(usize) -> Vec<f64> {
        move |id| {
            if id == 0 || id == total_states - 1 {
                return vec![0.0, 0.0];
            }
            vec![1.0, id as f64]
        }
    }
}
