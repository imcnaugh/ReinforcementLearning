mod deterministic;
mod stochastic;

pub use stochastic::StochasticPolicy;

pub use deterministic::DeterministicPolicy;

pub trait Policy {
    /// Returns a randomly chosen action id, for the given state id, based off the odds set for each
    /// action
    fn pick_action_for_state(&self, state_id: &str) -> Result<&str, String>;

    /// Returns a Vector of tuples, a f64 determining the odds of the action being chosen, and the
    /// String id of the action, for the state id that is given
    fn get_actions_for_state(&self, state_id: &str) -> Result<Vec<(f64, String)>, String>;
}
