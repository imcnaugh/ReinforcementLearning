mod action;
mod state;

pub use action::Actions;
pub use state::State;

fn iterative_policy_evaluation() -> f32 {
    0.0
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_iterative_policy_evaluation() {
        let mut initial_state = State::new();

        let state_a1_1 = State::new();
        let state_a1_2 = State::new();
        let state_a1_3 = State::new();

        let state_a2_1 = State::new();
        let state_a2_2 = State::new();
        let state_a2_3 = State::new();

        let mut action_1 = Actions::new();
        let mut action_2 = Actions::new();

        action_1.add_possible_next_state(0.5, &state_a1_1);
        action_1.add_possible_next_state(0.25, &state_a1_2);
        action_1.add_possible_next_state(0.25, &state_a1_3);

        action_2.add_possible_next_state(0.33, &state_a2_1);
        action_2.add_possible_next_state(0.33, &state_a2_2);
        action_2.add_possible_next_state(0.33, &state_a2_3);

        initial_state.add_action(&action_1);
        initial_state.add_action(&action_2);
    }
}
