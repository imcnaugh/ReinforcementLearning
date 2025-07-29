use rand::Rng;
use std::collections::HashMap;

/// # Reinforce Monte Carlo
///
/// The first policy gradient control method discussed in the book, but I still have some
/// reservations about how this could be scaled to
struct ReinforceMonteCarlo {
    preferences: HashMap<(String, String), f64>,
    learning_rate: f64,
    discount_rate: f64,
}

impl ReinforceMonteCarlo {
    pub fn new(learning_rate: f64, discount_rate: f64) -> Self {
        Self {
            preferences: HashMap::new(),
            learning_rate,
            discount_rate,
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

    /// Calculate the policy gradient for softmax policy
    /// ∇ ln π(a|s,θ) for softmax policy
    fn policy_gradient(
        &self,
        state_id: &str,
        actions: &[String],
        selected_action: &str,
    ) -> HashMap<String, f64> {
        let probabilities = self.softmax_probabilities(state_id, actions);
        let mut gradients = HashMap::new();

        for (i, action) in actions.iter().enumerate() {
            let prob = probabilities[i];
            let gradient = if action == selected_action {
                1.0 - prob // ∇ ln π(a|s,θ) = 1 - π(a|s,θ) for selected action
            } else {
                -prob // ∇ ln π(a|s,θ) = -π(a|s,θ) for non-selected actions
            };
            gradients.insert(action.clone(), gradient);
        }

        gradients
    }

    /// Generate an episode following the current policy
    /// Returns Vec<(state_id, action, reward)>
    fn generate_episode(&self) -> Vec<(String, String, f64)> {
        let mut episode = Vec::new();
        let mut current_state = generate_left_state(); // You'll need to adapt this to your environment

        // This is a simplified episode generation - you'll need to adapt this
        // to work with your specific environment/game
        loop {
            // Get available actions for current state
            if current_state.is_terminal() {
                break; // Terminal state
            }

            // Select action according to policy
            let action = match self.select_action_for_state(&current_state.get_id()) {
                Ok(a) => a,
                Err(_) => break,
            };

            // Take action and observe reward and next state
            let (reward, next_state) = &current_state.take_action(&action);

            episode.push((current_state.get_id().clone(), action, *reward));

            current_state = next_state.clone();
        }

        episode
    }

    /// Calculate returns G_t for each time step in the episode
    fn calculate_returns(&self, episode: &[(String, String, f64)]) -> Vec<f64> {
        let mut returns = vec![0.0; episode.len()];
        let mut g = 0.0;

        // Calculate returns backward from the end of episode
        for t in (0..episode.len()).rev() {
            g = episode[t].2 + self.discount_rate * g; // G_t = R_{t+1} + γ * G_{t+1}
            returns[t] = g;
        }

        returns
    }

    /// Update policy parameters using REINFORCE algorithm
    fn update_policy(&mut self, episode: &[(String, String, f64)], returns: &[f64]) {
        for (t, ((state_id, action, _reward), &g_t)) in
            episode.iter().zip(returns.iter()).enumerate()
        {
            // Get available actions for this state
            let actions = match state_id.clone().as_str() {
                "left" => generate_left_state().get_actions(),
                "center" => generate_center_state().get_actions(),
                "right" => generate_right_state().get_actions(),
                _ => panic!("Unknown state: {}", state_id),
            };

            // Calculate policy gradient ∇ ln π(A_t|S_t, θ)
            let gradients = self.policy_gradient(state_id, &actions, action);

            // Update preferences: θ ← θ + α * G_t * ∇ ln π(A_t|S_t, θ)
            for (act, gradient) in gradients {
                let key = (state_id.clone(), act.clone());
                let current_preference = self.preferences.get(&key).unwrap_or(&0.0);
                let update = self.learning_rate * g_t * gradient;
                self.preferences.insert(key, current_preference + update);
            }
        }
    }

    /// Main REINFORCE learning loop
    pub fn learn(&mut self, num_episodes: usize) {
        for episode_num in 0..num_episodes {
            // Generate an episode following π(·|·, θ)
            let episode = self.generate_episode();

            if episode.is_empty() {
                continue;
            }

            // Calculate returns G_t for each step
            let returns = self.calculate_returns(&episode);

            // Update policy parameters
            self.update_policy(&episode, &returns);

            if episode_num % 100 == 0 {
                println!(
                    "Episode {}: Episode length = {}",
                    episode_num,
                    episode.len()
                );
            }
        }
    }

    fn take_action(&self, state_id: &str, action: &str) -> (f64, String) {
        // Implement your environment dynamics
        // Returns (reward, next_state)
        match (state_id, action) {
            ("left", "move_center") => (0.0, "center".to_string()),
            ("center", "l") => (-1.0, "left".to_string()),
            ("center", "r") => (1.0, "right".to_string()),
            ("right", "move_center") => (0.0, "center".to_string()),
            _ => (0.0, "terminal".to_string()),
        }
    }

    fn is_terminal(&self, state_id: &str) -> bool {
        state_id == "terminal"
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
        let mut policy = ReinforceMonteCarlo::new(0.0, 0.0);
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
