use crate::chapter_04::State;
use std::sync::atomic::AtomicUsize;

pub struct Actions<'a> {
    id: String,
    reward: f32,
    possible_next_states: Vec<(f32, &'a State<'a>)>,
}

static mut NEXT_ACTION_ID: AtomicUsize = AtomicUsize::new(0);

impl<'a> Actions<'a> {
    pub fn new(reward: f32) -> Self {
        let next_action_id =
            unsafe { NEXT_ACTION_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst) };
        let next_action_id = next_action_id.to_string();

        Actions {
            id: next_action_id,
            reward,
            possible_next_states: Vec::new(),
        }
    }

    pub fn add_possible_next_state(&mut self, probability: f32, state: &'a State<'a>) {
        self.possible_next_states.push((probability, state));
    }

    pub fn get_reward(&self) -> f32 {
        self.reward
    }

    pub fn get_possible_next_states(&self) -> &Vec<(f32, &'a State<'a>)> {
        &self.possible_next_states
    }
}
