use crate::chapter_04::{Actions, State};
use std::sync::atomic::AtomicUsize;

pub struct Policy {
    id: String,
}

static mut NEXT_POLICY_ID: AtomicUsize = AtomicUsize::new(0);

impl Policy {
    pub fn new() -> Self {
        let next_policy_id =
            unsafe { NEXT_POLICY_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst) };
        let next_policy_id = next_policy_id.to_string();
        Policy { id: next_policy_id }
    }

    pub fn get_probabilities_for_each_action_of_state<'a>(
        &self,
        state: &'a State,
    ) -> Vec<(f32, &'a Actions)> {
        let num_of_actions = state.get_actions().len();
        let even_probabilities = 1f32 / num_of_actions as f32;
        state
            .get_actions()
            .iter()
            .map(|a| (even_probabilities, a))
            .collect::<Vec<(f32, &Actions)>>()
    }

    pub fn get_value_of_state(&self, state: &State, discount_rate: f32) -> f32 {
        self.get_probabilities_for_each_action_of_state(state)
            .iter()
            .map(|(prob, action)| prob * action.get_value(discount_rate))
            .sum()
    }
}
