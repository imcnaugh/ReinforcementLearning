use crate::chapter_04::State;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::AtomicUsize;

#[derive(Debug)]
pub struct Actions {
    id: String,
    possible_next_states: Vec<PossibleNextState>,
}

#[derive(Debug)]
pub struct PossibleNextState {
    probability: f32,
    state: Rc<RefCell<State>>,
    reward: f32,
}

static mut NEXT_ACTION_ID: AtomicUsize = AtomicUsize::new(0);

impl Actions {
    pub fn new() -> Self {
        let next_action_id =
            unsafe { NEXT_ACTION_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst) };
        let next_action_id = next_action_id.to_string();

        Actions {
            id: next_action_id,
            possible_next_states: Vec::new(),
        }
    }

    pub fn add_possible_next_state(
        &mut self,
        probability: f32,
        state: Rc<RefCell<State>>,
        reward: f32,
    ) {
        let possible_next_state = PossibleNextState::new(probability, state, reward);
        self.possible_next_states.push(possible_next_state);
    }

    pub fn get_possible_next_states(&self) -> &Vec<PossibleNextState> {
        &self.possible_next_states
    }

    pub fn get_value(&self, discount_rate: f32) -> f32 {
        self.possible_next_states
            .iter()
            .map(|p_ns| p_ns.get_value(discount_rate))
            .sum()
    }
}

impl PossibleNextState {
    fn new(probability: f32, state: Rc<RefCell<State>>, reward: f32) -> Self {
        PossibleNextState {
            probability,
            state,
            reward,
        }
    }

    pub fn get_probability(&self) -> f32 {
        self.probability
    }

    pub fn get_state(&self) -> &Rc<RefCell<State>> {
        &self.state
    }

    pub fn get_reward(&self) -> f32 {
        self.reward
    }

    pub fn get_value(&self, discount_rate: f32) -> f32 {
        self.probability * (self.reward + (self.state.borrow().get_value() * discount_rate))
    }
}
