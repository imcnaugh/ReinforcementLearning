use crate::chapter_04::Actions;
use std::sync::atomic::AtomicUsize;

pub struct State<'a> {
    id: String,
    value: f32,
    actions: Vec<&'a Actions<'a>>,
}

static mut NEXT_STATE_ID: AtomicUsize = AtomicUsize::new(0);

impl<'a> State<'a> {
    pub fn new() -> Self {
        let next_state_id =
            unsafe { NEXT_STATE_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst) };
        let next_state_id = next_state_id.to_string();

        State {
            id: next_state_id,
            value: 0.0,
            actions: Vec::new(),
        }
    }

    pub fn add_action(&mut self, action: &'a Actions<'a>) {
        self.actions.push(action);
    }

    pub fn get_value(&self) -> f32 {
        self.value
    }

    pub fn set_value(&mut self, value: f32) {
        self.value = value;
    }

    pub fn get_actions(&self) -> &Vec<&'a Actions<'a>> {
        &self.actions
    }

    pub fn get_id(&self) -> &String {
        &self.id
    }
}
