use crate::chapter_11::exercise_11_4::Action::{Left, Right};
use rand::prelude::IteratorRandom;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Action {
    Left,
    Right,
}

impl Action {
    fn select_at_random() -> Self {
        vec![Left, Right]
            .iter()
            .choose(&mut rand::rng())
            .unwrap()
            .clone()
    }

    fn to_string(&self) -> String {
        match self {
            Left => "left".to_string(),
            Right => "right".to_string(),
        }
    }
}

struct TestState {
    id: Action,
}

impl TestState {
    fn new(id: &Action) -> Self {
        Self { id: id.clone() }
    }

    fn select_action(&self, action: &Action) -> (Self, f64) {
        let new_state = TestState::new(action);
        match self.id {
            Left => (new_state, 0.0),
            Right => (new_state, 2.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_reward_output() {
        let mut state = TestState::new(&Left);
        for _ in 0..10 {
            let action = Action::select_at_random();
            let (new_state, reward) = state.select_action(&action);
            println!(
                "{} -> {} ({})",
                state.id.to_string(),
                action.to_string(),
                reward
            );
            state = new_state;
        }
    }
}
