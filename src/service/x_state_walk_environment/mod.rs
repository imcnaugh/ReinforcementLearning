use crate::attempts_at_framework::v2::state::State;
use std::error::Error;
use std::fmt::{Debug, Display, Formatter};

pub struct WalkStateFactory<VF>
where
    VF: Fn(usize) -> Vec<f64>,
{
    number_of_states: usize,
    walk_size: usize,
    value_function: VF,
}

pub struct WalkState<'a, VF>
where
    VF: Fn(usize) -> Vec<f64>,
{
    id: usize,
    factory: &'a WalkStateFactory<VF>,
    actions: Vec<String>,
    values: Vec<f64>,
    is_terminal: bool,
}

impl<VF> Clone for WalkState<'_, VF>
where
    VF: Fn(usize) -> Vec<f64>,
{
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

impl<VF> State for WalkState<'_, VF>
where
    VF: Fn(usize) -> Vec<f64>,
{
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

impl<VF> WalkStateFactory<VF>
where
    VF: Fn(usize) -> Vec<f64>,
{
    pub fn new(
        number_of_states: usize,
        walk_size: usize,
        value_function: VF,
    ) -> Result<Self, Box<WalkStateError>> {
        Ok(Self {
            number_of_states,
            walk_size,
            value_function,
        })
    }

    pub fn get_starting_state(&self) -> WalkState<VF> {
        let starting_point_id = (self.number_of_states / 2) as i32;
        self.generate_state_and_reward_for_id(starting_point_id).1
    }

    pub fn generate_state_and_reward_for_id(&self, id: i32) -> (f64, WalkState<VF>) {
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

        let values = (self.value_function)(new_id);

        let new_state = WalkState {
            id: new_id,
            factory: self,
            actions,
            values,
            is_terminal,
        };

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

    const DEFAULT_VALUE_FUNCTION: fn(usize) -> Vec<f64> = |id: usize| vec![id as f64];

    #[test]
    fn create_walk_state_factory() {
        let factory = WalkStateFactory::new(10, 2, DEFAULT_VALUE_FUNCTION).unwrap();
        assert_eq!(factory.number_of_states, 10);
        assert_eq!(factory.walk_size, 2);
    }

    #[test]
    fn create_walk_state() {
        let factory = WalkStateFactory::new(10, 2, DEFAULT_VALUE_FUNCTION).unwrap();
        let state = factory.get_starting_state();
        assert_eq!(state.id, 5);
        assert_eq!(state.actions.len(), 4);
        assert_eq!(state.values.len(), 1);
        assert_eq!(state.is_terminal, false);
    }

    #[test]
    fn take_action() {
        let factory = WalkStateFactory::new(10, 2, DEFAULT_VALUE_FUNCTION).unwrap();
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
        let factory = WalkStateFactory::new(10, 2, DEFAULT_VALUE_FUNCTION).unwrap();
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
