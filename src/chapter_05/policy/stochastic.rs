use crate::chapter_05::policy::Policy;
use rand::Rng;
use std::collections::HashMap;

pub struct StochasticPolicy {
    /// A map of the state id as a string, and a list that contains a tuple of
    ///  - f64: odds of taking action
    ///  - String: action id as a string
    state_action_probabilities: HashMap<String, Vec<(f64, String)>>,
}

impl StochasticPolicy {
    pub fn new() -> Self {
        StochasticPolicy {
            state_action_probabilities: HashMap::new(),
        }
    }

    pub fn set_state_action_probabilities(
        &mut self,
        state_id: &str,
        state_action_probabilities: Vec<(f64, String)>,
    ) -> Result<(), String> {
        let probability_sum = state_action_probabilities
            .iter()
            .fold(0.0, |acc, x| acc + x.0);

        if (probability_sum - 1.0).abs() > 0.0000001 {
            return Err(format!(
                "sum of probabilities expected to be 1.0, but was: {}",
                probability_sum
            ));
        }

        self.state_action_probabilities
            .insert(state_id.to_string(), state_action_probabilities);
        Ok(())
    }

    pub fn set_state_actions_probabilities_using_e_soft_probabilities(
        &mut self,
        state_id: &str,
        state_actions: Vec<(String)>,
        e: f64,
        best_action_id: String,
    ) -> Result<(), String> {
        if e < 0.0 || e > 1.0 {
            return Err(format!("e must be between 0.0 and 1.0, but was: {}", e));
        }

        if !state_actions.contains(&best_action_id) {
            return Err(format!(
                "best action id: {} not found in state actions: {:?}",
                best_action_id, state_actions
            ));
        }

        let best_action_probability = 1.0 - e + (e / state_actions.len() as f64);
        let other_action_probabilities = e / state_actions.len() as f64;

        let state_action_probabilities: Vec<(f64, String)> = state_actions
            .iter()
            .map(|action_id| {
                if *action_id == best_action_id {
                    (best_action_probability, action_id.to_string())
                } else {
                    (other_action_probabilities, action_id.to_string())
                }
            })
            .collect();

        self.set_state_action_probabilities(state_id, state_action_probabilities)
    }
}

impl Policy for StochasticPolicy {
    fn pick_action_for_state(&self, state_id: &str) -> Result<&str, String> {
        let mut random_number = rand::rng().random_range(0.0..1.0);

        if let Some(state_action_probabilities) = self.state_action_probabilities.get(state_id) {
            for (probability, action) in state_action_probabilities {
                if random_number <= *probability {
                    return Ok(action.as_str());
                }
                random_number = random_number - *probability;
            }
            Err(format!("error picking action for state id: {}", state_id))
        } else {
            Err(format!("no actions found for state id: {}", state_id))
        }
    }

    fn get_actions_for_state(&self, state_id: &str) -> Result<Vec<(f64, String)>, String> {
        match self.state_action_probabilities.get(state_id) {
            None => Err(format!("no actions found for state id: {}", state_id)),
            Some(state_and_actions) => Ok(state_and_actions.clone()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_action_from_stochastic_policy() {
        let mut stochastic_policy = StochasticPolicy::new();
        let state_id = "state_1";
        let state_action_probabilities = vec![(1.0, String::from("action_1"))];

        stochastic_policy
            .set_state_action_probabilities(state_id, state_action_probabilities)
            .expect("set_state_action_probabilities failed");

        assert_eq!(
            stochastic_policy.pick_action_for_state(state_id).unwrap(),
            "action_1"
        );
    }

    #[test]
    fn test_get_action_from_stochastic_policy_with_multiple_actions() {
        let mut stochastic_policy = StochasticPolicy::new();
        let state_id = "state_1";
        let state_action_probabilities = vec![
            (0.5, String::from("action_1")),
            (0.5, String::from("action_2")),
        ];

        stochastic_policy
            .set_state_action_probabilities(state_id, state_action_probabilities)
            .expect("set_state_action_probabilities failed");

        let action_id = stochastic_policy
            .pick_action_for_state(state_id)
            .expect("pick_action_for_state failed");
        println!("action_id: {}", action_id);
    }

    #[test]
    fn test_get_action_from_stochastic_policy_with_multiple_actions_and_sum_not_1() {
        let mut stochastic_policy = StochasticPolicy::new();
        let state_id = "state_1";
        let state_action_probabilities = vec![
            (0.5, String::from("action_1")),
            (0.25, String::from("action_2")),
        ];

        match stochastic_policy.set_state_action_probabilities(state_id, state_action_probabilities)
        {
            Ok(_) => panic!("set_state_action_probabilities should have failed"),
            Err(e) => assert_eq!(
                e,
                String::from("sum of probabilities expected to be 1.0, but was: 0.75")
            ),
        }
    }

    #[test]
    fn test_get_action_from_stochastic_policy_but_state_not_found() {
        let mut stochastic_policy = StochasticPolicy::new();
        let state_id = "state_1";

        match stochastic_policy.pick_action_for_state(state_id) {
            Ok(_) => panic!("pick_action_for_state should have failed"),
            Err(e) => assert_eq!(e, String::from("no actions found for state id: state_1")),
        }
    }

    #[test]
    fn test_get_actions_from_stochastic_policy_for_state() {
        let mut stochastic_policy = StochasticPolicy::new();
        let state_id = "state_1";
        let state_action_probabilities = vec![
            (0.2500, String::from("action_1")),
            (0.75, String::from("action_2")),
        ];

        stochastic_policy
            .set_state_action_probabilities(state_id, state_action_probabilities.clone())
            .expect("set_state_action_probabilities failed");

        match stochastic_policy.get_actions_for_state(state_id) {
            Ok(actions) => assert_eq!(actions, state_action_probabilities),
            Err(_) => panic!("get_actions_for_state failed"),
        }
    }

    #[test]
    fn test_get_actions_from_stochastic_policy_for_state_but_state_not_found() {
        let stochastic_policy = StochasticPolicy::new();
        let state_id = "state_1";

        match stochastic_policy.get_actions_for_state(state_id) {
            Ok(_) => panic!("get_actions_for_state should have failed"),
            Err(e) => assert_eq!(e, String::from("no actions found for state id: state_1")),
        }
    }

    #[test]
    fn test_set_state_actions_probabilities_using_e_soft_probabilities_2_actions() {
        let mut stochastic_policy = StochasticPolicy::new();
        let state_id = "state_1";
        let state_actions = vec![String::from("action_1"), String::from("action_2")];
        let e = 0.5;
        let best_action_id = String::from("action_1");

        stochastic_policy
            .set_state_actions_probabilities_using_e_soft_probabilities(
                state_id,
                state_actions,
                e,
                best_action_id,
            )
            .expect("set_state_actions_probabilities_using_e_soft_probabilities failed");
        let state_action_probabilities = stochastic_policy.get_actions_for_state(state_id).unwrap();
        assert_eq!(state_action_probabilities.len(), 2);
        assert!(state_action_probabilities.contains(&(0.75, String::from("action_1"))));
        assert!(state_action_probabilities.contains(&(0.25, String::from("action_2"))));
    }

    #[test]
    fn test_set_state_actions_probabilities_using_e_soft_probabilities_3_actions() {
        let mut stochastic_policy = StochasticPolicy::new();
        let state_id = "state_1";
        let state_actions = vec![
            String::from("action_1"),
            String::from("action_2"),
            String::from("action_3"),
        ];
        let e = 0.3;
        let best_action_id = String::from("action_2");

        stochastic_policy
            .set_state_actions_probabilities_using_e_soft_probabilities(
                state_id,
                state_actions,
                e,
                best_action_id,
            )
            .unwrap();

        let state_action_probabilities = stochastic_policy.get_actions_for_state(state_id).unwrap();
        assert_eq!(state_action_probabilities.len(), 3);
        let action_1_probability = state_action_probabilities
            .iter()
            .find(|x| x.1 == String::from("action_1"))
            .unwrap()
            .0;
        let action_2_probability = state_action_probabilities
            .iter()
            .find(|x| x.1 == String::from("action_2"))
            .unwrap()
            .0;
        let action_3_probability = state_action_probabilities
            .iter()
            .find(|x| x.1 == String::from("action_3"))
            .unwrap()
            .0;
        assert!((action_1_probability - 0.1).abs() < 0.0000001); // hooray floating point errors.
        assert!((action_2_probability - 0.8).abs() < 0.0000001);
        assert!((action_3_probability - 0.1).abs() < 0.0000001);
    }

    #[test]
    fn test_set_state_actions_probabilities_using_e_soft_probabilities_e_not_between_0_and_1() {
        let mut stochastic_policy = StochasticPolicy::new();
        let state_id = "state_1";
        let state_actions = vec![String::from("action_1"), String::from("action_2")];
        let e = 1.5;
        let best_action_id = String::from("action_1");

        match stochastic_policy.set_state_actions_probabilities_using_e_soft_probabilities(
            state_id,
            state_actions,
            e,
            best_action_id,
        ) {
            Ok(_) => panic!(
                "set_state_actions_probabilities_using_e_soft_probabilities should have failed"
            ),
            Err(e) => assert_eq!(
                e,
                String::from("e must be between 0.0 and 1.0, but was: 1.5")
            ),
        }
    }

    #[test]
    fn test_set_state_actions_probabilities_using_e_soft_probabilities_best_action_not_found() {
        let mut stochastic_policy = StochasticPolicy::new();
        let state_id = "state_1";
        let state_actions = vec![String::from("action_1"), String::from("action_2")];
        let e = 0.5;
        let best_action_id = String::from("action_3");

        match stochastic_policy.set_state_actions_probabilities_using_e_soft_probabilities(state_id, state_actions, e, best_action_id) {
            Ok(_) => panic!("set_state_actions_probabilities_using_e_soft_probabilities should have failed"),
            Err(e) => assert_eq!(e, String::from("best action id: action_3 not found in state actions: [\"action_1\", \"action_2\"]")),
        }
    }
}
