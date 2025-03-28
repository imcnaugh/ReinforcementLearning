use std::error::Error;
use std::fmt::{Debug, Display, Formatter};

pub trait Policy {
    fn select_action_for_state(&self, state_id: &str) -> Result<&str, Box<PolicyError>>;

    fn get_all_actions_for_state(&self, state_id: &str) -> Result<&Vec<String>, Box<PolicyError>>;
}

pub struct PolicyError {
    message: String,
}

impl PolicyError {
    pub fn new(message: String) -> Self {
        Self { message }
    }
}

impl Debug for PolicyError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "PolicyError: {}", self.message)
    }
}

impl Display for PolicyError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "PolicyError: {}", self.message)
    }
}

impl Error for PolicyError {}
