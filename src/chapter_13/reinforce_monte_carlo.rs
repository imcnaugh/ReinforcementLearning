use crate::attempts_at_framework::v1::policy::{Policy, PolicyError};

/// # Reinforce Monte Carlo
///
/// The first policy gradient control method discussed in the book, but I still have some
/// reservations about how this could be scaled to
struct ReinforceMonteCarlo {
    parameter_step_size: f64,
}

impl ReinforceMonteCarlo {
    pub fn new(parameter_step_size: f64) -> Self {
        Self {
            parameter_step_size,
        }
    }
}

impl Policy for ReinforceMonteCarlo {
    fn select_action_for_state(&self, state_id: &str) -> Result<String, Box<PolicyError>> {
        todo!()
    }
}
