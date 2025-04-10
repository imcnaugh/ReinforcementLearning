use crate::attempts_at_framework::v1::policy::policy::{Policy, PolicyError};
use crate::attempts_at_framework::v1::policy::DeterministicPolicy;
use rand::prelude::IndexedRandom;
use std::collections::HashMap;

#[derive(Debug, Clone)]
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

    pub fn get_actions_for_state(&self, state_id: &str) -> Option<&Vec<(String, f64)>> {
        self.state_action_odds.get(state_id)
    }

    pub fn to_deterministic_policy(&self) -> DeterministicPolicy {
        let mut deterministic_policy = DeterministicPolicy::new();
        self.state_action_odds
            .iter()
            .for_each(|(state_id, actions_and_odds)| {
                let best_action = actions_and_odds
                    .iter()
                    .max_by(|x, y| {
                        let x_v = x.1;
                        let y_v = y.1;
                        x_v.partial_cmp(&y_v).unwrap()
                    })
                    .unwrap()
                    .0
                    .clone();
                deterministic_policy.set_actions_for_state(state_id.clone(), best_action);
            });
        deterministic_policy
    }
}

impl Policy for StochasticPolicy {
    fn select_action_for_state(&self, state_id: &str) -> Result<String, Box<PolicyError>> {
        match self.state_action_odds.get(state_id) {
            None => Err(Box::new(PolicyError::new(format!(
                "state id {} not found",
                state_id
            )))),
            Some(actions_and_odds) => {
                if actions_and_odds.is_empty() {
                    return Err(Box::new(PolicyError::new(format!(
                        "state id {} has no actions",
                        state_id
                    ))));
                }

                let mut rng = rand::rng();

                match actions_and_odds
                    .choose_weighted(&mut rng, |a| a.1)
                    .map(|a| a.0.clone())
                {
                    Ok(a) => Ok(a),
                    Err(_) => Err(Box::new(PolicyError::new(format!(
                        "state id {} not found",
                        state_id
                    )))),
                }
            }
        }
    }
}
