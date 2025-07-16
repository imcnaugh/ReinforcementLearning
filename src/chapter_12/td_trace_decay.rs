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
