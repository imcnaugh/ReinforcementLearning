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
) -> Model
where
    S: State,
    P: Policy,
{
    let mut model_builder = ModelBuilder::new();
    model_builder
        .set_loss_function(Box::new(MeanSquaredError))
        .set_input_size(starting_state.get_values().len());

    // model_builder.add_layer(LayerBuilder::new(LayerType::LINEAR, 2));
    // model_builder.add_layer(LayerBuilder::new(LayerType::RELU, 2));
    model_builder.add_layer(LayerBuilder::new(LayerType::LINEAR, 1));

    let mut model = model_builder.build().unwrap();

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

            if let Some((reward, values)) = queue.pop_back().unwrap() {
                let expected = sliding_reward_total
                    + (discount_rate.powi(n as i32) * model.predict(current_state.get_values())[0]);
                model.train(values, vec![expected], learning_rate);

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
        let value_function = generate_normalized_value_function(number_of_states);

        let state_factory = WalkStateFactory::new(number_of_states, 100, &value_function).unwrap();

        let starting_state = state_factory.get_starting_state();

        let learned_model = n_step_td_ann(
            starting_state,
            RandomPolicy::new(),
            discount_rate,
            n,
            episode_count,
            learning_rate,
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

    fn generate_normalized_value_function(total_states: usize) -> impl Fn(WalkState) -> Vec<f64> {
        move |state| {
            let mut response = vec![0.0, 0.0];
            if !state.is_terminal() {
                response[0] = 1.0;
                response[1] = state.get_id().parse::<usize>().unwrap() as f64 / total_states as f64;
            }
            response
        }
    }
}
