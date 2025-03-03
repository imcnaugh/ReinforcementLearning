mod action;
mod state;

pub use action::Actions;
pub use state::State;
use std::cell::RefCell;
use std::ops::DerefMut;
use std::rc::Rc;

fn iterative_policy_evaluation(
    states: &mut Vec<Rc<RefCell<State>>>,
    discount_rate: f32,
    threshold: f32,
) {
    let mut delta: f32 = 0.0;
    let mut iteration = 0;

    // Loop until convergence (when the change in value is smaller than the threshold)
    loop {
        delta = 0.0;
        iteration += 1;

        // Iterate through each state and update its value
        for mut state in states.iter_mut() {
            let v_old = state.borrow().get_value(); // Save the old value for delta computation
            let mut new_value: f32 = 0.0;

            // Iterate through each action in the state's action list
            for action in state.borrow().get_actions() {
                let mut action_value = 0.0;

                // Calculate the expected value for each possible next state and reward
                for (probability, next_state) in action.get_possible_next_states() {
                    action_value += probability
                        * (action.get_reward() + discount_rate * next_state.borrow().get_value());
                }

                new_value = new_value.max(action_value); // Take the action that maximizes the value
            }

            // Update the state's value and calculate the maximum change (delta)
            state.borrow_mut().set_value(new_value);
            delta = delta.max((v_old - state.borrow().get_value()).abs());
        }

        // Stop if the maximum change in value (delta) is smaller than the threshold
        if delta < threshold {
            println!("Policy Evaluation converged after {} iterations", iteration);
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    #[test]
    fn test_iterative_policy_evaluation() {
        let mut s1 = Rc::new(RefCell::new(State::new()));
        let mut s2 = Rc::new(RefCell::new(State::new()));
        let mut s3 = Rc::new(RefCell::new(State::new()));
        let s_terminal = Rc::new(RefCell::new(State::new()));

        let mut a1 = Actions::new(1_f32);
        let mut a2 = Actions::new(0_f32);
        let mut a3 = Actions::new(-1_f32);
        let mut a4 = Actions::new(5_f32);

        a1.add_possible_next_state(1_f32, s_terminal.clone());
        a2.add_possible_next_state(0.75, s2.clone());
        a2.add_possible_next_state(0.25, s3.clone());
        a3.add_possible_next_state(1_f32, s_terminal.clone());
        a4.add_possible_next_state(1_f32, s_terminal.clone());

        s1.borrow_mut().add_action(a1);
        s1.borrow_mut().add_action(a2);
        s2.borrow_mut().add_action(a3);
        s3.borrow_mut().add_action(a4);

        let mut states = vec![s1, s2, s3];

        println!("before");
        for state in states.iter() {
            println!(
                "state id: {}, value: {}",
                state.borrow().get_id(),
                state.borrow().get_value()
            )
        }

        iterative_policy_evaluation(&mut states, 0.9, 0.0001);
        println!("after");
        for state in states.iter() {
            println!(
                "state id: {}, value: {}",
                state.borrow().get_id(),
                state.borrow().get_value()
            )
        }
    }
}
