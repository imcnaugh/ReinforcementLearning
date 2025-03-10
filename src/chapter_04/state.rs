use crate::chapter_04::Action;
use std::sync::atomic::AtomicUsize;

#[derive(Debug)]
pub struct State {
    id: String,
    value: f32,
    actions: Vec<Action>,
    debug_value_arr: Vec<f32>,
}

static mut NEXT_STATE_ID: AtomicUsize = AtomicUsize::new(0);

impl State {
    pub fn new() -> Self {
        let next_state_id =
            unsafe { NEXT_STATE_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst) };
        let next_state_id = next_state_id.to_string();

        State {
            id: next_state_id,
            value: 0.0,
            actions: Vec::new(),
            debug_value_arr: Vec::new(),
        }
    }

    pub fn add_action(&mut self, action: Action) {
        self.actions.push(action);
    }

    pub fn get_value(&self) -> f32 {
        self.value
    }

    pub fn set_value(&mut self, value: f32) {
        self.debug_value_arr.push(self.value);
        self.value = value;
    }

    pub fn get_actions(&self) -> &Vec<Action> {
        &self.actions
    }

    pub fn get_id(&self) -> &String {
        &self.id
    }

    pub fn get_value_to_max_action_value(&self, discount_rate: f32) -> f32 {
        let mut max_action_value = f32::MIN;
        self.actions.iter().for_each(|action| {
            let action_value = action.get_value(discount_rate);
            if action_value > max_action_value {
                max_action_value = action_value;
            }
        });
        max_action_value
    }

    pub fn get_debug_value_arr(&self) -> &Vec<f32> {
        &self.debug_value_arr
    }
}
