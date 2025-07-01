use rand::prelude::IndexedRandom;

#[derive(Clone)]
struct State {
    id: String,
    next_states: Vec<(f64, State)>,
    true_value: f64,
}

impl State {
    fn new(id: String, next_states: Vec<(f64, State)>, true_value: f64) -> State {
        State {
            id,
            next_states,
            true_value,
        }
    }

    /// Randomly selects and returns a reference to one of the possible next state transitions
    /// Returns a tuple containing (reward, next_state)
    fn transition(&self) -> &(f64, State) {
        self.next_states.choose(&mut rand::rng()).unwrap()
    }

    fn is_terminal(&self) -> bool {
        self.next_states.is_empty()
    }

    fn get_true_value(&self) -> f64 {
        self.true_value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_mean_square_td_error_setup() {
        let terminal_state_b = State::new("terminal_b".to_string(), vec![], 0.0);
        let terminal_state_c = State::new("terminal_c".to_string(), vec![], 0.0);

        let state_b = State::new("b".to_string(), vec![(1.0, terminal_state_b)], 1.0);
        let state_c = State::new("c".to_string(), vec![(0.0, terminal_state_c)], 0.0);
        let state_a = State::new("a".to_string(), vec![(0.0, state_b), (0.0, state_c)], 0.5);

        let transition_a = state_a.transition();
        assert_eq!(transition_a.0, 0.0);
        match transition_a.1.id.as_str() {
            "b" => {
                let next = transition_a.1.transition();
                assert_eq!(next.0, 1.0);
                assert_eq!(next.1.id, "terminal_b");
            }
            "c" => {
                let next = transition_a.1.transition();
                assert_eq!(next.0, 0.0);
                assert_eq!(next.1.id, "terminal_c");
            }
            _ => panic!("Unexpected state"),
        }
    }

    #[test]
    fn monte_carlo_method_for_value() {
        let terminal_state_b = State::new("terminal_b".to_string(), vec![], 0.0);
        let terminal_state_c = State::new("terminal_c".to_string(), vec![], 0.0);

        let state_b = State::new("b".to_string(), vec![(1.0, terminal_state_b)], 1.0);
        let state_c = State::new("c".to_string(), vec![(0.0, terminal_state_c)], 0.0);
        let state_a = State::new(
            "a".to_string(),
            vec![(0.0, state_b.clone()), (0.0, state_c.clone())],
            0.5,
        );

        let episode_count = 1000000;
        let learning_rate = 0.001;
        let mut state_a_value = 0.0;
        let mut state_b_value = 0.0;
        let mut state_c_value = 0.0;

        for _ in 0..episode_count {
            let mut history = vec![];

            let mut current_state = state_a.clone();

            while !current_state.is_terminal() {
                let curr_id = current_state.id.clone();
                let (reward, next_state) = current_state.transition();
                let his = (reward.clone(), curr_id);
                history.push(his);
                current_state = next_state.clone();
            }

            let mut total_reward = 0.0;
            for (reward, state_id) in history.iter().rev() {
                total_reward += reward;
                match state_id.as_str() {
                    "a" => state_a_value += learning_rate * (total_reward - state_a_value),
                    "b" => state_b_value += learning_rate * (total_reward - state_b_value),
                    "c" => state_c_value += learning_rate * (total_reward - state_c_value),
                    _ => panic!("Unexpected state"),
                }
            }
        }

        println!(
            "State A value: {}, State B value: {}, State C value: {}",
            state_a_value, state_b_value, state_c_value
        );

        fn in_acceptable_error(a: f64, b: f64) -> bool {
            (a - b).abs() < 0.01
        }

        assert!(in_acceptable_error(state_a_value, state_a.get_true_value()));
        assert!(in_acceptable_error(state_b_value, state_b.get_true_value()));
        assert!(in_acceptable_error(state_c_value, state_c.get_true_value()));
    }
}
