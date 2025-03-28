use std::collections::HashMap;
use crate::attempts_at_framework::v1::policy::policy::{Policy, PolicyError};

pub struct DeterministicPolicy {
    state_action_map: HashMap<String, String>,
}

impl DeterministicPolicy {
    pub fn new() -> Self {
        Self {
            state_action_map: HashMap::new(),
        }
    }
}

impl Policy for DeterministicPolicy {
    fn select_action_for_state(&self, state_id: &str) -> Result<&str, Box<PolicyError>> {
        match self.state_action_map.get(state_id) {
            None => Err(Box::new(PolicyError::new(format!("state id {} not found", state_id)))),
            Some(action) => Ok(action),
        }
    }

    fn get_all_actions_for_state(&self, state_id: &str) -> Result<&Vec<String>, Box<PolicyError>> {
        match self.state_action_map.get(state_id) {
            None => Err(Box::new(PolicyError::new(format!("state id {} not found", state_id)))),
            Some(action) => Ok(&vec![action.to_string()]),
        }
    }

    fn set_actions_for_state(&mut self, state_id: &str, actions: Vec<String>) {
        if actions.len() != 1 {
            panic!("deterministic policy only supports one action per state");
        }
        self.state_action_map.insert(state_id.to_string(), actions[0].to_string());
    }
}