/// # State
///
/// defines simple functions that any state should be able to implement.
pub trait State {

    /// returns the id of the state, this needs to be unique to the state
    fn get_id(&self) -> String;

    /// returns a list of actions available from this state
    fn get_actions(&self) -> Vec<String>;

    /// Is this state terminal, as in, is this the last state of the episode.
    fn is_terminal(&self) -> bool;

    /// Given the id of an action, execute that action, and return the reward for that action
    /// and the next state.
    fn take_action(&self, action: &str) -> (f64, Self);
}