use crate::attempts_at_framework::v2::state::State;
use std::error::Error;
use std::fmt::{Debug, Display, Formatter};

pub struct WalkStateFactory {
    number_of_states: usize,
    walk_size: usize,
    group_size: usize,
}

#[derive(Clone)]
pub struct WalkState<'a> {
    id: usize,
    factory: &'a WalkStateFactory,
    actions: Vec<String>,
    values: Vec<f64>,
    is_terminal: bool,
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

impl WalkStateFactory {
    pub fn new(
        number_of_states: usize,
        group_size: usize,
        walk_size: usize,
    ) -> Result<Self, Box<WalkStateError>> {
        if number_of_states < group_size {
            return Err(Box::new(WalkStateError {
                msg: format!(
                    "number_of_states ({}) must be greater then group_size ({})",
                    number_of_states, group_size
                ),
            }));
        }
        if number_of_states % group_size != 0 {
            return Err(Box::new(WalkStateError {
                msg: format!(
                    "number_of_states ({}) must be divisible by group_size ({})",
                    number_of_states, group_size
                ),
            }));
        }
        Ok(Self {
            number_of_states,
            walk_size,
            group_size,
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

        let actions: Vec<String> = (0..self.walk_size)
            .flat_map(|i| vec![format!("{}", i), format!("-{}", i)])
            .collect();
        let new_id = id.clamp(0, self.number_of_states as i32 - 1) as usize;

        let values = self.state_aggregation_values_for_state_id(new_id);

        let new_state = WalkState {
            id: new_id,
            factory: self,
            actions,
            values,
            is_terminal,
        };

        (reward, new_state)
    }

    fn state_aggregation_values_for_state_id(&self, new_id: usize) -> Vec<f64> {
        let mut groups = vec![0.0; self.number_of_states / self.group_size];
        let group_id = new_id / self.group_size;
        groups[group_id] = 1.0;
        groups
    }

    fn basic_values_for_state_id(&self, new_id: usize) -> Vec<f64> {
        vec![-1.0, new_id as f64]
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

    #[test]
    fn create_walk_state_factory() {
        let factory = WalkStateFactory::new(10, 2, 2).unwrap();
        assert_eq!(factory.number_of_states, 10);
        assert_eq!(factory.walk_size, 2);
        assert_eq!(factory.group_size, 2);
    }

    #[test]
    fn create_walk_state() {
        let factory = WalkStateFactory::new(10, 2, 2).unwrap();
        let state = factory.get_starting_state();
        assert_eq!(state.id, 5);
        assert_eq!(state.actions.len(), 4);
        assert_eq!(state.values.len(), 5);
        assert_eq!(state.is_terminal, false);
    }

    #[test]
    fn take_action() {
        let factory = WalkStateFactory::new(10, 2, 2).unwrap();
        let state = factory.get_starting_state();
        let (reward, new_state) = state.take_action("1");
        assert_eq!(reward, 0.0);
        assert_eq!(new_state.id, 6);
        assert_eq!(new_state.actions.len(), 4);
        assert_eq!(new_state.values.len(), 5);
        assert_eq!(new_state.is_terminal, false);
    }

    #[test]
    fn test_rewards_on_ends() {
        let factory = WalkStateFactory::new(10, 2, 2).unwrap();
        let (reward, state) = factory.generate_state_and_reward_for_id(0);
        assert_eq!(reward, -1.0);
        assert_eq!(state.id, 0);
        assert_eq!(state.actions.len(), 4);
        assert_eq!(state.values.len(), 5);
        assert_eq!(state.is_terminal, true);
        let (reward, state) = factory.generate_state_and_reward_for_id(9);
        assert_eq!(reward, 1.0);
        assert_eq!(state.id, 9);
        assert_eq!(state.actions.len(), 4);
        assert_eq!(state.values.len(), 5);
        assert_eq!(state.is_terminal, true);
    }
}
