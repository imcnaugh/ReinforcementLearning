use crate::chapter_04::{Action, State};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
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

    fn get_probabilities_for_each_action_of_state<'a>(
        &self,
        state: &'a State,
    ) -> Vec<(f32, &'a Action)> {
        let possible_actions = state.get_actions();
        let action_values = possible_actions
            .iter()
            .map(|a| -> (f32, &Action) {
                let value = a.get_value(1.0);
                (value, a)
            })
            .collect::<Vec<(f32, &Action)>>();

        let max_value = &action_values
            .iter()
            .map(|(value, _)| *value)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);

        let max_actions = action_values
            .iter()
            .filter(|(value, _)| (*value - max_value).abs() < f32::EPSILON)
            .collect::<Vec<&(f32, &Action)>>();
        let max_actions_count = &max_actions.len();
        let action_probabilities = 1f32 / *max_actions_count as f32;

        max_actions
            .iter()
            .map(|(_, a)| (action_probabilities, *a))
            .collect::<Vec<(f32, &Action)>>()
    }
}

pub struct MutablePolicy {
    id: String,
    state_and_action_probabilities: HashMap<String, Vec<(f32, String)>>,
}

impl MutablePolicy {
    pub fn new(states: &Vec<Rc<RefCell<State>>>) -> Self {
        let next_policy_id =
            unsafe { NEXT_POLICY_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst) };
        let next_policy_id = next_policy_id.to_string();

        let state_and_action_probabilities: HashMap<String, Vec<(f32, String)>> = states
            .iter()
            .map(|s| {
                let s = s.borrow();
                let prob = 1_f32 / s.get_actions().len() as f32;
                let value = s
                    .get_actions()
                    .iter()
                    .map(|a| -> (f32, String) { (prob, a.get_id().to_string()) })
                    .collect::<Vec<(f32, String)>>();
                (s.get_id().clone(), value)
            })
            .collect();

        MutablePolicy {
            id: next_policy_id,
            state_and_action_probabilities,
        }
    }

    pub fn get_optimal_action_for_state<'a>(&self, state: &'a State) -> &'a Action {
        let mut max_value = f32::MIN;
        let mut max_action_ids = String::new();
        state.get_actions().iter().for_each(|a| {
            let value = a.get_value(1.0);
            if value > max_value {
                max_action_ids = a.get_id().to_string();
                max_value = value;
            }
        });
        let action = state
            .get_actions()
            .iter()
            .find(|a| a.get_id() == max_action_ids)
            .unwrap();
        action
    }

    pub fn converge(
        &mut self,
        mut states: Vec<Rc<RefCell<State>>>,
        discount_rate: f32,
        threshold: f32,
    ) {
        let mut delta: f32 = 0.0;
        let mut iteration = 0;
        println!("starting policy convergence with threshold: {}", threshold);

        loop {
            println!("starting to estimate state value for policy");
            loop {
                delta = 0.0;
                iteration += 1;

                // Iterate through each state and update its value
                for mut state in states.iter_mut() {
                    let v_old = state.borrow().get_value(); // Save the old value for delta computation
                    let new_value: f32 =
                        <dyn Policy>::get_value_of_state(self, &state.borrow(), discount_rate);

                    // Update the state's value and calculate the maximum change (delta)
                    state.borrow_mut().set_value(new_value);
                    delta = delta.max((v_old - new_value).abs());
                }

                if delta < threshold {
                    break;
                }
            }
            println!("finished estimating state value for policy");
            println!("starting to update policy");

            let mut policy_stable = true;
            for state in states.iter() {
                let mut max_value = f32::MIN;
                let mut max_action_ids = Vec::new();

                state.borrow().get_actions().iter().for_each(|a| {
                    let value = a.get_value(discount_rate);
                    if value > max_value {
                        max_action_ids = vec![a.get_id().to_string()];
                        max_value = value;
                    } else if value == max_value {
                        max_action_ids.push(a.get_id().to_string());
                    }
                });

                let expected_odds: f32 = 1.0_f32 / (max_action_ids.len() as f32);

                let new_probs: Vec<(f32, String)> = self
                    .state_and_action_probabilities
                    .get(state.borrow().get_id())
                    .unwrap()
                    .iter()
                    .map(|(prob, a_id)| {
                        if max_action_ids.contains(&a_id) {
                            if prob.clone() != expected_odds {
                                policy_stable = false;
                            }
                            (expected_odds, a_id.to_string())
                        } else {
                            if prob.clone() != 0.0 {
                                policy_stable = false;
                            }
                            (0.0_f32, a_id.to_string())
                        }
                    })
                    .collect();

                self.state_and_action_probabilities
                    .insert(state.borrow().get_id().clone(), new_probs);
            }

            println!(
                "finished updating policy, policy is stable: {}",
                policy_stable
            );

            if policy_stable {
                break;
            }
        }
    }
}

impl Policy for MutablePolicy {
    fn get_id(&self) -> &str {
        self.id.as_str()
    }

    fn get_probabilities_for_each_action_of_state<'a>(
        &self,
        state: &'a State,
    ) -> Vec<(f32, &'a Action)> {
        state
            .get_actions()
            .iter()
            .map(|a| {
                // TODO refactor, this is gross but it should work
                let prob = self
                    .state_and_action_probabilities
                    .get(state.get_id())
                    .unwrap()
                    .iter()
                    .find(|(_, id)| id == a.get_id())
                    .unwrap()
                    .0;
                (prob, a)
            })
            .collect()
    }
}
