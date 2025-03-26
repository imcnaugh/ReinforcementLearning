use crate::chapter_05::policy::Policy;
use crate::chapter_05::race_track::state::State;
use crate::chapter_06::blackjack_test_state::BlackJackTestState;

pub fn value_function<P: Policy>(starting_state: BlackJackTestState, policy: &P, size_step_parameter: f64, discount_rate: f64) -> Vec<BlackJackTestState> {
    let mut states: Vec<BlackJackTestState> = vec![starting_state];
    let mut state = &mut states[0];
    while state.is_terminal() == false {
        let action = policy.pick_action_for_state(&state.get_id()).unwrap();
        let (reward, new_state) = state.take_action(&action);
        let error = reward + (discount_rate * new_state.get_state_value()) - state.get_state_value();
        let new_state_value = state.get_state_value() + (size_step_parameter * error);
        &state.set_state_value(new_state_value);
        states.push(new_state);
        state = states.last_mut().unwrap();
    };
    states
}

#[cfg(test)]
mod tests {
    use crate::chapter_05::policy::DeterministicPolicy;
    use super::*;

    #[test]
    fn test_value_function() {
        let starting_state = BlackJackTestState::new(10, 10, false, false);

        let mut policy = DeterministicPolicy::new();

        (10..17).for_each(|player_count| {
            policy.set_action_for_state(format!("{}_{}_{}", player_count, false, 10).as_str(), "hit");
            policy.set_action_for_state(format!("{}_{}_{}", player_count, true, 10).as_str(), "hit");
        });
        (17..=21).for_each(|player_count| {
            policy.set_action_for_state(format!("{}_{}_{}", player_count, false, 10).as_str(), "stand");
            policy.set_action_for_state(format!("{}_{}_{}", player_count, true, 10).as_str(), "stand");
        });

        let states = value_function(starting_state, &policy, 0.5, 1.0);

        for state in states {
            println!("state id: {} has value: {}", state.get_id(), state.get_state_value());
        }
    }
}
