use crate::attempts_at_framework::v1::policy::{Policy, PolicyError};
use std::collections::HashMap;

pub struct PolicyGradient {
    preferences: HashMap<(String, String), f64>,
}

impl Policy for PolicyGradient {
    fn select_action_for_state(&self, state_id: &str) -> Result<String, Box<PolicyError>> {
        todo!()
    }
}
