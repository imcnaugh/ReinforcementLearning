use rand::Rng;
use std::collections::HashMap;

/// # Reinforce Monte Carlo
///
/// The first policy gradient control method discussed in the book, but I still have some
/// reservations about how this could be scaled to
struct ReinforceMonteCarlo {
    parameter_step_size: f64,
    preferences: HashMap<(String, String), f64>,
}

impl ReinforceMonteCarlo {
    pub fn new(parameter_step_size: f64) -> Self {
        Self {
            parameter_step_size,
            preferences: HashMap::new(),
        }
    }

    fn get_preference(&self, state_id: &str, action: &str) -> f64 {
        *self
            .preferences
            .get(&(state_id.to_string(), action.to_string()))
            .unwrap_or(&0.0)
    }

    fn softmax_probabilities(&self, state_id: &str, actions: &[String]) -> Vec<f64> {
        let preferences: Vec<f64> = actions
            .iter()
            .map(|a| self.get_preference(state_id, a))
            .collect();
        let max_pref = preferences
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let exp_prefs: Vec<f64> = preferences.iter().map(|&p| (p - max_pref).exp()).collect();
        let sum_exp: f64 = exp_prefs.iter().cloned().sum();
        exp_prefs.iter().map(|&p| p / sum_exp).collect()
    }
}

impl Policy for ReinforceMonteCarlo {
    fn select_action_for_state(&self, state_id: &str) -> Result<String, Box<PolicyError>> {
        let state = match state_id {
            "left" => generate_left_state(),
            "center" => generate_center_state(),
            "right" => generate_right_state(),
            _ => panic!("Unknown state: {}", state_id),
        };

        let actions = state.get_actions();
        let probabilities = self.softmax_probabilities(state_id, &actions);

        let mut cum_prob = 0.0;
        let sample: f64 = rand::rng().random();

        for (action, &prob) in actions.iter().zip(probabilities.iter()) {
            cum_prob += prob;
            if sample <= cum_prob {
                return Ok(action.to_string());
            }
        }

        Ok(actions.last().unwrap().to_string())
    }
}

use crate::attempts_at_framework::v1::policy::{Policy, PolicyError};
use crate::attempts_at_framework::v2::state::State;
use crate::chapter_13::example_13_1::{
    generate_center_state, generate_left_state, generate_right_state,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "This test is not deterministic"]
    fn test_select_action_for_state() {
        let mut policy = ReinforceMonteCarlo::new(0.1);
        policy
            .preferences
            .insert(("center".to_string(), "l".to_string()), 0.9);
        policy
            .preferences
            .insert(("center".to_string(), "r".to_string()), 0.1);
        let state = generate_center_state();
        let action = policy.select_action_for_state("center").unwrap();
        assert_eq!(action, "l");
    }
}
