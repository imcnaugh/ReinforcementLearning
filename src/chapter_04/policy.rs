use std::collections::HashMap;
use crate::chapter_04::{Action, State};
use std::sync::atomic::AtomicUsize;

pub trait Policy {
    fn get_id(&self) -> &str;

    fn get_probabilities_for_each_action_of_state<'a>(
        &self,
        state: &'a State,
    ) -> Vec<(f32, &'a Action)>;
}

static mut NEXT_POLICY_ID: AtomicUsize = AtomicUsize::new(0);

impl dyn Policy + '_ {
    pub fn get_value_of_state(&self, state: &State, discount_rate: f32) -> f32 {
        self.get_probabilities_for_each_action_of_state(state)
            .iter()
            .map(|(prob, action)| prob * action.get_value(discount_rate))
            .sum()
    }
}

pub struct RandomPolicy {
    id: String,
}

impl RandomPolicy {
    pub fn new() -> Self {
        let next_policy_id =
            unsafe { NEXT_POLICY_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst) };
        let next_policy_id = next_policy_id.to_string();
        RandomPolicy { id: next_policy_id }
    }
}

impl Policy for RandomPolicy {
    fn get_id(&self) -> &str {
        self.id.as_str()
    }

    fn get_probabilities_for_each_action_of_state<'a>(
        &self,
        state: &'a State,
    ) -> Vec<(f32, &'a Action)> {
        let num_of_actions = state.get_actions().len();
        let even_probabilities = 1f32 / num_of_actions as f32;
        state
            .get_actions()
            .iter()
            .map(|a| (even_probabilities, a))
            .collect::<Vec<(f32, &Action)>>()
    }
}

pub struct GreedyPolicy {
    id: String,
}

impl GreedyPolicy {
    pub fn new() -> Self {
        let next_policy_id =
            unsafe { NEXT_POLICY_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst) };
        let next_policy_id = next_policy_id.to_string();
        GreedyPolicy { id: next_policy_id }
    }
}

impl Policy for GreedyPolicy {
    fn get_id(&self) -> &str {
        self.id.as_str()
    }

    fn get_probabilities_for_each_action_of_state<'a>(&self, state: &'a State) -> Vec<(f32, &'a Action)> {
        let possible_actions = state.get_actions();
        let action_values = possible_actions.iter().map(|a| -> (f32, &Action) {
            let value = a.get_value(1.0);
            (value, a)
        }).collect::<Vec<(f32, &Action)>>();

        let max_value = &action_values.iter().map(|(value, _)| *value).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(0.0);

        let max_actions = action_values
            .iter()
            .filter(|(value, _)| (*value - max_value).abs() < f32::EPSILON)
            .collect::<Vec<&(f32, &Action)>>();
        let max_actions_count = &max_actions.len();
        let action_probabilities = 1f32 / *max_actions_count as f32;

        max_actions.iter().map(|(_, a)| (action_probabilities, *a)).collect::<Vec<(f32, &Action)>>()
    }
}

pub struct MutablePolicy {
    id: String,
    state_and_action_probabilities: HashMap<String, Vec<(f32, String)>>,
}

impl MutablePolicy {
    pub fn new(states: Vec<&State>) -> Self {
        let next_policy_id =
            unsafe { NEXT_POLICY_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst) };
        let next_policy_id = next_policy_id.to_string();

        let state_and_action_probabilities: HashMap<String, Vec<(f32, String)>> = states.iter().map(|s| {
            let prob = 1_f32 / s.get_actions().len() as f32;
            let value = s.get_actions().iter().map(|a| -> (f32, String) {
                (prob, a.get_id().to_string())
            }).collect::<Vec<(f32, String)>>();
            (s.get_id().clone(), value)
        }).collect();

        MutablePolicy {
            id: next_policy_id,
            state_and_action_probabilities,
        }
    }
}

impl Policy for MutablePolicy {
    fn get_id(&self) -> &str {
        self.id.as_str()
    }

    fn get_probabilities_for_each_action_of_state<'a>(&self, state: &'a State) -> Vec<(f32, &'a Action)> {
        state.get_actions().iter().map(|a| {
            // TODO refactor, this is gross but it should work
            let prob = self.state_and_action_probabilities.get(state.get_id()).unwrap().iter().find(|(_, id)| id == a.get_id()).unwrap().0;
            (prob, a)
        }).collect()
    }
}