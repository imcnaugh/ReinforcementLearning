use crate::attempts_at_framework::v1::policy::{EGreedyPolicy, Policy};
use crate::attempts_at_framework::v1::state::State;
use rand::prelude::IteratorRandom;
use std::collections::HashMap;

pub fn tabular_dyna_q<S: State>(
    iteration_count: usize,
    states: Vec<S>,
    discount_rate: f64,
    size_step_parameter: f64,
    n: usize,
) {
    let state_map: HashMap<String, S> = states.iter().map(|s| (s.get_id(), s.clone())).collect();
    let mut state_action_values: HashMap<String, f64> = HashMap::new();
    let mut model: HashMap<(String, String), (f64, String)> = HashMap::new();
    let mut policy: EGreedyPolicy = EGreedyPolicy::new(0.1);
    let mut rng = rand::rng();

    (0..iteration_count).for_each(|_| {
        let state = states.iter().choose(&mut rng).unwrap().clone();
        let action = match policy.select_action_for_state(&state.get_id()) {
            Ok(a) => a,
            Err(_) => state.get_actions().iter().choose(&mut rng).unwrap().clone(),
        };

        let state_action_id = get_state_action_id(&state.get_id(), &action);
        let (reward, next_state) = state.take_action(&action);
        let current_state_action_value = state_action_values.get(&state_action_id).unwrap_or(&0.0);
        let next_state_best_action_value =
            get_max_value_of_state_actions(&state_action_values, &next_state).unwrap_or(0.0);
        let new_state_action_value = current_state_action_value
            + (size_step_parameter
                * (reward + (discount_rate * next_state_best_action_value)
                    - current_state_action_value));
        state_action_values.insert(state_action_id.clone(), new_state_action_value);
        model.insert(
            (state.get_id(), action),
            (new_state_action_value, next_state.get_id()),
        );

        (0..n).for_each(|_| {
            let ((s, a), (r, ns)) = model.iter().choose(&mut rng).unwrap();
            let s_a_id = get_state_action_id(s, a);
            let current_s_a_value = state_action_values.get(&s_a_id).unwrap_or(&0.0);
            let ns = state_map.get(ns).unwrap().clone();
            let best_ns_value =
                get_max_value_of_state_actions(&state_action_values, &ns).unwrap_or(0.0);
            let new_s_a_value = current_s_a_value
                + (size_step_parameter * (r + (discount_rate * best_ns_value) - current_s_a_value));
            state_action_values.insert(s_a_id.clone(), new_s_a_value);
        });
    });
}

fn get_max_value_of_state_actions<S: State>(
    state_action_values: &HashMap<String, f64>,
    state: &S,
) -> Option<f64> {
    let actions = state.get_actions();
    let mut max_value: Option<f64> = None;
    for action in actions {
        let state_action_id = get_state_action_id(&state.get_id(), &action);
        if let Some(value) = state_action_values.get(&state_action_id) {
            match max_value {
                None => max_value = Some(value.clone()),
                Some(current_max) => {
                    max_value = Some(current_max.max(value.clone()));
                }
            }
        }
    }
    max_value
}

fn get_state_action_id(state_id: &str, action_id: &str) -> String {
    format!("{}_{}", state_id, action_id)
}
