mod action;
mod policy;
mod state;

use crate::chapter_04::policy::Policy;
pub use action::Actions;
pub use state::State;
use std::cell::RefCell;
use std::ops::{Deref, DerefMut};
use std::rc::Rc;

fn iterative_policy_evaluation(
    policy: &Policy,
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
            let mut new_value: f32 = f32::NEG_INFINITY;

            // Iterate through each action in the state's action list
            for (action_probability, action) in
                policy.get_probabilities_for_each_action_of_state(&state.borrow())
            {
                let mut action_value = 0.0;

                // Calculate the expected value for each possible next state and reward
                for (next_state_probability, next_state) in action.get_possible_next_states() {
                    action_value += action_probability
                        * (next_state_probability
                            * (action.get_reward()
                                + discount_rate * next_state.borrow().get_value()));
                }

                new_value = new_value.max(action_value); // Take the action that maximizes the value
            }

            // Update the state's value and calculate the maximum change (delta)
            state.borrow_mut().set_value(new_value);
            delta = delta.max((v_old - new_value).abs());
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
    use crate::chapter_04::policy::Policy;
    use crate::service::{ChartBuilder, ChartData};
    use plotters::prelude::{BLUE, GREEN, RED};
    use std::cell::RefCell;
    use std::path::PathBuf;
    use std::sync::atomic::AtomicUsize;

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
        a2.add_possible_next_state(0.35, s2.clone());
        a2.add_possible_next_state(0.65, s3.clone());
        a3.add_possible_next_state(1_f32, s_terminal.clone());
        a4.add_possible_next_state(1_f32, s_terminal.clone());

        s1.borrow_mut().add_action(a1);
        s1.borrow_mut().add_action(a2);
        s2.borrow_mut().add_action(a3);
        s3.borrow_mut().add_action(a4);

        let mut states = vec![s1, s2, s3];

        let simple_policy = Policy::new();

        println!("before");
        for state in states.iter() {
            println!(
                "state id: {}, value: {}",
                state.borrow().get_id(),
                state.borrow().get_value()
            )
        }

        iterative_policy_evaluation(&simple_policy, &mut states, 0.9, 0.0000001);
        println!("after");
        for state in states.iter() {
            println!(
                "state id: {}, value: {}",
                state.borrow().get_id(),
                state.borrow().get_value()
            )
        }

        let mut styles = vec![BLUE.into(), RED.into(), GREEN.into()]
            .into_iter()
            .cycle();

        let mut chart_builder = ChartBuilder::new();
        chart_builder
            .set_path(PathBuf::from("output/chapter4/iterativeConversion.png"))
            .set_x_label("Iterations".to_string())
            .set_y_label("Estimated State Value".to_string())
            .set_title("Iterative Policy Evaluation".to_string());

        let mut next_id: AtomicUsize = AtomicUsize::new(1);
        for state in states {
            let points = state
                .borrow()
                .get_debug_value_arr()
                .iter()
                .enumerate()
                .map(|(i, v)| -> (f32, f32) { (i as f32, v.clone()) })
                .collect::<Vec<(f32, f32)>>();
            let next_id = next_id.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let data = ChartData::new(format!("State {}", next_id), points, styles.next().unwrap());
            chart_builder.add_data(data);
        }

        chart_builder.create_chart().unwrap();
    }
}
