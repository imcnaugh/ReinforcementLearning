use rand::prelude::IndexedRandom;

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
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
