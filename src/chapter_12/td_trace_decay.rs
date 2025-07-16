use crate::attempts_at_framework::v2::state::State;
use rand::prelude::IteratorRandom;

struct TdTraceDecay<S: State> {
    starting_states: Vec<S>,
    weights: Vec<f64>,
    discount_rate: f64,
    trace_decay_rate: f64,
    learning_rate: f64,
}

impl<S: State> TdTraceDecay<S> {
    pub fn new(
        starting_states: Vec<S>,
        discount_rate: f64,
        trace_decay_rate: f64,
        learning_rate: f64,
    ) -> Self {
        if starting_states.is_empty() {
            panic!("TDTraceDecay cannot be called with empty starting_states");
        }

        let weight_count = starting_states[0].get_values().len();

        Self {
            starting_states,
            weights: vec![0.0; weight_count],
            discount_rate,
            trace_decay_rate,
            learning_rate,
        }
    }

    pub fn learn_for_episode(&mut self) {
        let rand = &mut rand::rng();
        let mut trace_decay_vector = vec![0.0; self.weights.len()];

        let mut current_state = self.starting_states.iter().choose(rand).unwrap().clone();

        while !current_state.is_terminal() {
            let actions = current_state.get_actions();
            let action = actions.iter().choose(rand).unwrap();
            let (reward, next_state) = current_state.take_action(action);

            let discounted_trace_decay_rate = self.discount_rate * self.trace_decay_rate;
            trace_decay_vector = trace_decay_vector
                .iter()
                .zip(current_state.get_values())
                .map(|(trace_decay, gradient_decent)| {
                    discounted_trace_decay_rate * trace_decay + gradient_decent
                })
                .collect();

            let error = reward + self.get_state_value_estimate(&next_state)
                - self.get_state_value_estimate(&current_state);

            self.weights = self
                .weights
                .iter()
                .zip(&trace_decay_vector)
                .map(|(w, t)| w + self.learning_rate * error * t)
                .collect();
            current_state = next_state;
        }
    }

    fn get_state_value_estimate(&self, state: &S) -> f64 {
        state
            .get_values()
            .iter()
            .zip(&self.weights)
            .map(|(s, w)| s * w)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::service::x_state_walk_environment::{WalkState, WalkStateFactory};

    #[test]
    fn do_random_walk() {
        let discount_rate = 1.0;
        let trace_decay_rate = 0.5;
        let learning_rate = 0.00001;

        let number_of_states = 100;
        // let value_function = generate_polynomial_value_function(number_of_states, 1);
        let value_function = generate_simple_value_function(number_of_states);
        let state_factory = WalkStateFactory::new(number_of_states, 10, &value_function).unwrap();

        let idk = state_factory.get_starting_state();
        let starting_states = vec![idk];

        let mut td_trace_decay = TdTraceDecay::new(
            starting_states,
            discount_rate,
            trace_decay_rate,
            learning_rate,
        );

        for _ in 0..1000 {
            td_trace_decay.learn_for_episode();
        }

        for id in 0..number_of_states {
            let (_, state) = state_factory.generate_state_and_reward_for_id(id as i32);
            let value = td_trace_decay.get_state_value_estimate(&state);
            println!("State id: {}, has value: {}", id, value);
        }

        println!("{:?}", td_trace_decay.weights)
    }

    fn generate_simple_value_function(total_states: usize) -> impl Fn(WalkState) -> Vec<f64> {
        move |state| {
            let state_id_as_usize = state.get_id().parse::<usize>().unwrap();
            if state_id_as_usize == 0 || state_id_as_usize == total_states - 1 {
                return vec![0.0, 0.0];
            }
            vec![1.0, state_id_as_usize as f64]
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
}
