use crate::chapter_04::Action;
use std::sync::atomic::AtomicUsize;

#[derive(Debug)]
pub struct State {
    id: String,
    capital: Option<i32>,
    value: f32,
    actions: Vec<Action>,
    debug_value_arr: Vec<f32>,
    is_terminal: bool,
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
            capital: None,
            actions: Vec::new(),
            debug_value_arr: Vec::new(),
            is_terminal: false,
        }
    }

    pub fn set_id(&mut self, id: String) {
        self.id = id;
    }

    pub fn set_is_terminal(&mut self, is_terminal: bool) {
        self.is_terminal = is_terminal;
    }

    pub fn get_capital(&self) -> Option<i32> {
        self.capital
    }

    pub fn set_capital(&mut self, capital: i32) {
        self.capital = Some(capital);
    }

    pub fn get_is_terminal(&self) -> bool {
        self.is_terminal
    }

    pub fn add_action(&mut self, action: Action) {
        self.actions.push(action);
    }

    pub fn get_value(&self) -> f32 {
        if self.is_terminal {
            0_f32
        } else {
            self.value
        }
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

    pub fn get_max_action_description(&self, discount_rate: f32) -> String {
        if self.is_terminal {
            return String::new();
        }
        let mut max_action_value = f32::MIN;
        let mut max_action_description = String::new();
        self.actions.iter().for_each(|action| {
            let action_value = action.get_value(discount_rate);
            if action_value > max_action_value {
                max_action_value = action_value;
                max_action_description = action.get_description().unwrap().to_string();
            }
        });
        max_action_description
    }

    pub fn get_max_action_value(&self, discount_rate: f32) -> f32 {
        if self.is_terminal {
            return 0_f32;
        }
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
