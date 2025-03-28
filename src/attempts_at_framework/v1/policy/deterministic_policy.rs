use crate::attempts_at_framework::v1::policy::policy::{Policy, PolicyError};
use std::collections::HashMap;

pub struct DeterministicPolicy {
    state_action_map: HashMap<String, String>,
}

impl DeterministicPolicy {
    pub fn new() -> Self {
        Self {
            state_action_map: HashMap::new(),
        }
    }

    pub fn set_actions_for_state(&mut self, state_id: String, action: String) {
        self.state_action_map.insert(state_id, action);
    }
}

impl Policy for DeterministicPolicy {
    fn select_action_for_state(&self, state_id: &str) -> Result<String, Box<PolicyError>> {
        match self.state_action_map.get(state_id) {
            None => Err(Box::new(PolicyError::new(format!(
                "state id {} not found",
                state_id
            )))),
            Some(action) => Ok(action.clone()),
        }
    }
}
