use crate::attempts_at_framework::v2::state::State;
use std::error::Error;
use std::fmt::{Debug, Display, Formatter};

pub struct WalkStateFactory<'a> {
    number_of_states: usize,
    walk_size: usize,
    value_function: &'a dyn Fn(WalkState) -> Vec<f64>,
}

pub struct WalkState<'a> {
    id: usize,
    factory: &'a WalkStateFactory<'a>,
    actions: Vec<String>,
    values: Vec<f64>,
    is_terminal: bool,
}

impl Clone for WalkState<'_> {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            factory: self.factory,
            actions: self.actions.clone(),
            values: self.values.clone(),
            is_terminal: self.is_terminal,
        }
    }
}

impl State for WalkState<'_> {
    fn get_id(&self) -> String {
        self.id.to_string()
    }

    fn get_actions(&self) -> Vec<String> {
        self.actions.clone()
    }

    fn is_terminal(&self) -> bool {
        self.is_terminal
    }

    fn take_action(&self, action: &str) -> (f64, Self) {
        let new_id = self.id as i32 + action.parse::<i32>().unwrap();
        self.factory.generate_state_and_reward_for_id(new_id)
    }

    fn get_values(&self) -> Vec<f64> {
        self.values.clone()
    }
}

impl<'a> WalkStateFactory<'a> {
    pub fn new(
        number_of_states: usize,
        walk_size: usize,
        value_function: &'a dyn Fn(WalkState) -> Vec<f64>,
    ) -> Result<Self, Box<WalkStateError>> {
        Ok(Self {
            number_of_states,
            walk_size,
            value_function,
        })
    }

    pub fn get_starting_state(&self) -> WalkState {
        let starting_point_id = (self.number_of_states / 2) as i32;
        self.generate_state_and_reward_for_id(starting_point_id).1
    }

    pub fn generate_state_and_reward_for_id(&self, id: i32) -> (f64, WalkState) {
        let (reward, is_terminal) = if id <= 0 {
            (-1.0, true)
        } else if id >= self.number_of_states as i32 - 1 {
            (1.0, true)
        } else {
            (0.0, false)
        };

        let actions: Vec<String> = (1..=self.walk_size)
            .flat_map(|i| vec![format!("{}", i), format!("-{}", i)])
            .collect();
        let new_id = id.clamp(0, self.number_of_states as i32 - 1) as usize;

        let mut new_state = WalkState {
            id: new_id,
            factory: self,
            actions,
            values: Vec::new(),
            is_terminal,
        };

        let values = (self.value_function)(new_state.clone());
        new_state.values = values;

        (reward, new_state)
    }
}

#[derive(Debug)]
pub struct WalkStateError {
    msg: String,
}

impl Display for WalkStateError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "WalkStateError: {}", self.msg)
    }
}

impl Error for WalkStateError {}

#[cfg(test)]
mod tests {
    use super::*;

    const DEFAULT_VALUE_FUNCTION: fn(WalkState) -> Vec<f64> = |s| vec![s.id as f64];

    #[test]
    fn create_walk_state_factory() {
        let factory = WalkStateFactory::new(10, 2, &DEFAULT_VALUE_FUNCTION).unwrap();
        assert_eq!(factory.number_of_states, 10);
        assert_eq!(factory.walk_size, 2);
    }

    #[test]
    fn create_walk_state() {
        let factory = WalkStateFactory::new(10, 2, &DEFAULT_VALUE_FUNCTION).unwrap();
        let state = factory.get_starting_state();
        assert_eq!(state.id, 5);
        assert_eq!(state.actions.len(), 4);
        assert_eq!(state.values.len(), 1);
        assert_eq!(state.is_terminal, false);
    }

    #[test]
    fn take_action() {
        let factory = WalkStateFactory::new(10, 2, &DEFAULT_VALUE_FUNCTION).unwrap();
        let state = factory.get_starting_state();
        let (reward, new_state) = state.take_action("1");
        assert_eq!(reward, 0.0);
        assert_eq!(new_state.id, 6);
        assert_eq!(new_state.actions.len(), 4);
        assert_eq!(new_state.values.len(), 1);
        assert_eq!(new_state.is_terminal, false);
    }

    #[test]
    fn test_rewards_on_ends() {
        let factory = WalkStateFactory::new(10, 2, &DEFAULT_VALUE_FUNCTION).unwrap();
        let (reward, state) = factory.generate_state_and_reward_for_id(0);
        assert_eq!(reward, -1.0);
        assert_eq!(state.id, 0);
        assert_eq!(state.actions.len(), 4);
        assert_eq!(state.values.len(), 1);
        assert_eq!(state.is_terminal, true);
        let (reward, state) = factory.generate_state_and_reward_for_id(9);
        assert_eq!(reward, 1.0);
        assert_eq!(state.id, 9);
        assert_eq!(state.actions.len(), 4);
        assert_eq!(state.values.len(), 1);
        assert_eq!(state.is_terminal, true);
    }
}
