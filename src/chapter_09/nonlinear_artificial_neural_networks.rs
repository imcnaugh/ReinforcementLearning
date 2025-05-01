use std::collections::VecDeque;
use rand::prelude::IteratorRandom;
use crate::attempts_at_framework::v1::policy::Policy;
use crate::attempts_at_framework::v2::artificial_neural_network::loss_functions::mean_squared_error::MeanSquaredError;
use crate::attempts_at_framework::v2::artificial_neural_network::model::Model;
use crate::attempts_at_framework::v2::artificial_neural_network::model::model_builder::{LayerBuilder, LayerType, ModelBuilder};
use crate::attempts_at_framework::v2::state::State;

pub fn n_step_td_ann<S, P>(
    starting_state: S,
    policy: P,
    discount_rate: f64,
    n: usize,
    episode_count: usize,
    learning_rate: f64,
    layers: Vec<LayerBuilder>,
) -> Model
where
    S: State,
    P: Policy,
{
    let mut model_builder = ModelBuilder::new();
    model_builder
        .set_loss_function(Box::new(MeanSquaredError))
        .set_input_size(starting_state.get_values().len());

    layers.into_iter().for_each(|l| {
        model_builder.add_layer(l);
    });

    let mut model = model_builder.build().unwrap();

    (0..episode_count).for_each(|_| {
        let mut current_state = starting_state.clone();
        let mut queue: VecDeque<(S, f64)> = VecDeque::new();
        let mut rewards: VecDeque<f64> = VecDeque::new();

        while !current_state.is_terminal() {
            // Take action and get next state and reward
            let action = select_action(&current_state, &policy);
            let (reward, next_state) = current_state.take_action(&action);

            // Store state and reward
            queue.push_back((current_state.clone(), reward));
            rewards.push_back(reward);

            // Update n-step return when we have enough samples
            if queue.len() >= n {
                let (old_state, _) = queue.pop_front().unwrap();
                let mut n_step_return = 0.0;

                // Calculate n-step return
                for (i, r) in rewards.iter().enumerate() {
                    n_step_return += discount_rate.powi(i as i32) * r;
                }

                // Add bootstrap value if not at terminal state
                if !next_state.is_terminal() {
                    n_step_return +=
                        discount_rate.powi(n as i32) * model.predict(next_state.get_values())[0];
                }

                // Train the model
                model.train(old_state.get_values(), vec![n_step_return], learning_rate);

                rewards.pop_front();
            }

            current_state = next_state;
        }

        // Handle remaining states in queue
        while !queue.is_empty() {
            let (old_state, _) = queue.pop_front().unwrap();
            let mut n_step_return = 0.0;

            for (i, r) in rewards.iter().enumerate() {
                n_step_return += discount_rate.powi(i as i32) * r;
            }

            model.train(old_state.get_values(), vec![n_step_return], learning_rate);

            if !rewards.is_empty() {
                rewards.pop_front();
            }
        }
    });

    model
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

#[cfg(test)]
mod tests {
    use crate::attempts_at_framework::v1::policy::RandomPolicy;
    use crate::attempts_at_framework::v2::artificial_neural_network::model::model_builder::{
        LayerBuilder, LayerType,
    };
    use crate::attempts_at_framework::v2::state::State;
    use crate::chapter_09::nonlinear_artificial_neural_networks::n_step_td_ann;
    use crate::service::x_state_walk_environment::{WalkState, WalkStateFactory};
    use crate::service::{LineChartBuilder, LineChartData};
    use plotters::prelude::{ShapeStyle, BLUE, RED};
    use std::path::PathBuf;

    #[test]
    fn random_walk_n_step_td_ann() {
        let number_of_states = 1000;
        let discount_rate = 1.0;
        let learning_rate = 0.0001;
        let episode_count = 10000;
        let n = 10;

        // let value_function = generate_state_aggregation_value_function(number_of_states, 100);
        // let value_function = generate_normalized_value_function(number_of_states);
        let value_function = generate_simple_value_function(number_of_states);

        let layers = vec![
            // LayerBuilder::new(LayerType::LINEAR, 1),
            // LayerBuilder::new(LayerType::RELU, 2),
            LayerBuilder::new(LayerType::LINEAR, 1),
        ];

        let state_factory = WalkStateFactory::new(number_of_states, 100, &value_function).unwrap();

        let starting_state = state_factory.get_starting_state();

        let learned_model = n_step_td_ann(
            starting_state,
            RandomPolicy::new(),
            discount_rate,
            n,
            episode_count,
            learning_rate,
            layers,
        );

        let data_points: Vec<(f32, f32)> = (0..number_of_states)
            .map(|i| {
                let state = state_factory.generate_state_and_reward_for_id(i as i32).1;
                let value = learned_model.predict(state.get_values())[0];
                (i as f32, value as f32)
            })
            .collect();

        let mut line_chart_builder = LineChartBuilder::new();
        line_chart_builder.set_path(PathBuf::from(
            "output/chapter9/thousand_state_random_walk-n_step_ann.png",
        ));
        line_chart_builder.set_title(format!(
            "Thousand state random walk {} step - Neural Network",
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

        learned_model.print_weights();
    }

    fn generate_state_aggregation_value_function(
        total_states: usize,
        group_size: usize,
    ) -> impl Fn(WalkState) -> Vec<f64> {
        let number_of_groups = total_states / group_size;

        move |state| {
            let group_id = state.get_id().parse::<usize>().unwrap() / group_size;
            let mut response = vec![0.0; number_of_groups];
            response[group_id] = 1.0;
            response
        }
    }

    fn generate_normalized_value_function(total_states: usize) -> impl Fn(WalkState) -> Vec<f64> {
        move |state| {
            let mut response = vec![0.0, 0.0];
                response[0] = 1.0;
                response[1] = state.get_id().parse::<usize>().unwrap() as f64 / total_states as f64;
            response
        }
    }

    fn generate_simple_value_function(total_states: usize) -> impl Fn(WalkState) -> Vec<f64> {
        move |state| {vec![
                    (state.get_id().parse::<usize>().unwrap() + 1) as f64 / total_states as f64,
                ]
            }
    }
}
