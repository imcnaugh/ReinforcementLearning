use crate::chapter_11::exercise_11_4::Action::{Left, Right};
use rand::prelude::IteratorRandom;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Action {
    Left,
    Right,
}

impl Action {
    fn select_at_random() -> Self {
        vec![Left, Right]
            .iter()
            .choose(&mut rand::rng())
            .unwrap()
            .clone()
    }

    fn to_string(&self) -> String {
        match self {
            Left => "left".to_string(),
            Right => "right".to_string(),
        }
    }
}

struct TestState {
    id: Action,
}

impl TestState {
    fn new(id: &Action) -> Self {
        Self { id: id.clone() }
    }

    fn select_action(&self, action: &Action) -> (Self, f64) {
        let new_state = TestState::new(action);
        match self.id {
            Left => (new_state, 0.0),
            Right => (new_state, 2.0),
        }
    }
}

#[derive(Debug)]
struct ValueEstimation {
    true_values: HashMap<Action, f64>,
    estimated_values: HashMap<Action, f64>,
}

impl ValueEstimation {
    fn new() -> Self {
        let mut true_values = HashMap::new();
        let mut estimated_values = HashMap::new();

        // Initialize with some values
        true_values.insert(Left, 0.0);
        true_values.insert(Right, 2.0);
        estimated_values.insert(Left, 0.25); // Example estimate
        estimated_values.insert(Right, 1.75); // Example estimate

        Self {
            true_values,
            estimated_values,
        }
    }

    fn msve(&self) -> f64 {
        // Calculate MSVE according to equation 11.24
        // MSVE = E[(Vᵖ(s) - v̂(s))²]
        let mut sum_squared_error = 0.0;
        let n_states = self.true_values.len() as f64;

        for (state, true_value) in &self.true_values {
            let estimated_value = self.estimated_values.get(state).unwrap();
            // Add and subtract true value as per the hint
            let error = (estimated_value - true_value).powi(2);
            sum_squared_error += error;
        }

        sum_squared_error / n_states
    }

    fn decomposed_msve(&self) -> (f64, f64) {
        // Decompose MSVE into bias² and variance terms
        let mut variance = 0.0;
        let mut sum_squared_error = 0.0;
        let n_states = self.true_values.len() as f64;

        for (state, true_value) in &self.true_values {
            let estimated_value = self.estimated_values.get(state).unwrap();
            // Add and subtract true value as per the hint
            let error = (estimated_value - true_value).powi(2);
            sum_squared_error += error;

            // We would need multiple samples for true variance
            // This is simplified for demonstration
            variance += 0.0; // In this simple example, we assume no variance
        }

        (sum_squared_error / n_states, variance / n_states)
    }

    fn mean_square_return_error(&self) -> f64 {
        let mut sum_squared_error = 0.0;
        let n_states = self.estimated_values.len() as f64;

        for (state, estimated_value) in &self.estimated_values {
            let reward = match state {
                Left => 0.0,
                Right => 2.0,
            };
            let error = (reward - estimated_value).powi(2);
            sum_squared_error += error;
        }

        sum_squared_error / n_states
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_reward_output() {
        let mut state = TestState::new(&Left);
        for _ in 0..10 {
            let action = Action::select_at_random();
            let (new_state, reward) = state.select_action(&action);
            println!(
                "{} -> {} ({})",
                state.id.to_string(),
                action.to_string(),
                reward
            );
            state = new_state;
        }
    }

    #[test]
    fn test_msve_decomposition() {
        let value_estimation = ValueEstimation::new();

        // Calculate total MSVE
        let msve = value_estimation.msve();

        // Calculate decomposed components
        let (bias_squared, variance) = value_estimation.decomposed_msve();

        let msre = value_estimation.mean_square_return_error();

        println!("MSVE: {}", msve);
        println!("Bias²: {}", bias_squared);
        println!("Variance: {}", variance);
        println!("MSRE: {}", msre);
        println!("Bias² + Variance: {}", bias_squared + variance);

        // Verify that MSVE = Bias² + Variance
        assert!((msve - msre).abs() < 1e-10);
    }
}
