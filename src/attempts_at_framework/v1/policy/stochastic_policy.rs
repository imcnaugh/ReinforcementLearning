use crate::attempts_at_framework::v1::policy::policy::{Policy, PolicyError};
use rand::prelude::IndexedRandom;
use std::collections::HashMap;

pub struct StochasticPolicy {
    state_action_odds: HashMap<String, Vec<(String, f64)>>,
}

impl StochasticPolicy {
    pub fn new() -> Self {
        Self {
            state_action_odds: HashMap::new(),
        }
    }

    pub fn set_actions_for_state(
        &mut self,
        state_id: String,
        actions_and_odds: Vec<(String, f64)>,
    ) {
        self.state_action_odds.insert(state_id, actions_and_odds);
    }
}

impl Policy for StochasticPolicy {
    fn select_action_for_state(&self, state_id: &str) -> Result<&str, Box<PolicyError>> {
        match self.state_action_odds.get(state_id) {
            None => Err(Box::new(PolicyError::new(format!(
                "state id {} not found",
                state_id
            )))),
            Some(actions_and_odds) => {
                let mut rng = rand::rng();

                match actions_and_odds
                    .choose_weighted(&mut rng, |a| a.1)
                    .map(|a| a.0)
                {
                    Ok(a) => Ok(&a),
                    Err(_) => Err(Box::new(PolicyError::new(format!(
                        "state id {} not found",
                        state_id
                    )))),
                }
            }
        }
    }

    fn get_all_actions_for_state(&self, state_id: &str) -> Result<&Vec<String>, Box<PolicyError>> {
        match self.state_action_odds.get(state_id) {
            None => Err(Box::new(PolicyError::new(format!(
                "state id {} not found",
                state_id
            )))),
            Some(actions_and_odds) => {
                let actions = actions_and_odds.iter().map(|a| a.0).collect();
                Ok(&actions)
            }
        }
    }
}
