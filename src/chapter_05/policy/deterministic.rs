use crate::chapter_05::policy::Policy;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct DeterministicPolicy {
    state_action_map: HashMap<String, String>,
}

impl DeterministicPolicy {
    pub fn new() -> Self {
        DeterministicPolicy {
            state_action_map: HashMap::new(),
        }
    }

    pub fn set_action_for_state(&mut self, state_id: &str, action_id: &str) {
        self.state_action_map
            .insert(state_id.to_string(), action_id.to_string());
    }
}

impl Policy for DeterministicPolicy {
    fn pick_action_for_state(&self, state_id: &str) -> Result<&str, String> {
        match self.state_action_map.get(state_id) {
            None => Err(format!("no actions found for state id: {}", state_id)),
            Some(action_id) => Ok(action_id),
        }
    }

    fn get_actions_for_state(&self, state_id: &str) -> Result<Vec<(f64, String)>, String> {
        match self.state_action_map.get(state_id) {
            None => Err(format!("no actions found for state id: {}", state_id)),
            Some(action) => Ok(vec![(1.0, action.clone())]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_action_from_deterministic_policy() {
        let mut deterministic_policy = DeterministicPolicy::new();
        let state_id = "state_1";
        let action_id = "action_1";

        deterministic_policy.set_action_for_state(state_id, action_id);

        assert_eq!(
            deterministic_policy
                .pick_action_for_state(state_id)
                .unwrap(),
            action_id
        );
    }

    #[test]
    fn test_get_action_from_deterministic_policy_but_state_not_found() {
        let deterministic_policy = DeterministicPolicy::new();
        let state_id = "state_1";

        match deterministic_policy.pick_action_for_state(state_id) {
            Ok(_) => panic!("pick_action_for_state should have failed"),
            Err(e) => assert_eq!(e, String::from("no actions found for state id: state_1")),
        }
    }

    #[test]
    fn test_get_actions_from_deterministic_policy_for_state() {
        let mut deterministic_policy = DeterministicPolicy::new();
        let state_id = "state_1";
        let action_id = "action_1";

        deterministic_policy.set_action_for_state(state_id, action_id);

        match deterministic_policy.get_actions_for_state(state_id) {
            Ok(actions) => assert_eq!(actions, vec![(1.0, action_id.to_string())]),
            Err(_) => panic!("get_actions_for_state failed"),
        }
    }

    #[test]
    fn test_get_actions_from_deterministic_policy_for_state_but_state_not_found() {
        let deterministic_policy = DeterministicPolicy::new();
        let state_id = "state_1";

        match deterministic_policy.get_actions_for_state(state_id) {
            Ok(_) => panic!("get_actions_for_state should have failed"),
            Err(e) => assert_eq!(e, String::from("no actions found for state id: state_1")),
        }
    }

    #[test]
    fn test_set_action_for_state() {
        let mut deterministic_policy = DeterministicPolicy::new();
        let state_id = "state_1";
        let action_id_1 = "action_1";

        deterministic_policy.set_action_for_state(state_id, action_id_1);
        assert_eq!(
            deterministic_policy.state_action_map.get(state_id).unwrap(),
            action_id_1
        );

        let action_id_2 = "action_2";

        deterministic_policy.set_action_for_state(state_id, action_id_2);
        assert_eq!(
            deterministic_policy.state_action_map.get(state_id).unwrap(),
            action_id_2
        );
    }
}
