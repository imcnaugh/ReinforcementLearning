use crate::attempts_at_framework::v1::policy::policy::{Policy, PolicyError};
use rand::prelude::IndexedRandom;
use std::collections::HashMap;

pub struct RandomPolicy {
    state_actions: HashMap<String, Vec<String>>,
}

impl RandomPolicy {
    pub fn new() -> Self {
        Self {
            state_actions: HashMap::new(),
        }
    }

    pub fn set_actions_for_state(&mut self, state_id: String, actions: Vec<String>) {
        self.state_actions.insert(state_id, actions);
    }
}

impl Policy for RandomPolicy {
    fn select_action_for_state(&self, state_id: &str) -> Result<&str, Box<PolicyError>> {
        let actions = self.get_all_actions_for_state(state_id)?;

        let mut rng = rand::rng();
        let action = actions.choose(&mut rng).unwrap();

        Ok(action)
    }

    fn get_all_actions_for_state(&self, state_id: &str) -> Result<&Vec<String>, Box<PolicyError>> {
        match self.state_actions.get(state_id) {
            None => Err(Box::new(PolicyError::new(format!(
                "state id {} not found",
                state_id
            )))),
            Some(actions) => Ok(actions),
        }
    }
}
