mod action;
mod policy;
mod state;

use crate::chapter_04::policy::{Policy, RandomPolicy};
pub use action::Action;
pub use state::State;
use std::cell::RefCell;
use std::f64::consts::E;
use std::ops::{Deref, DerefMut};
use std::rc::Rc;

fn iterative_policy_evaluation(
    policy: &dyn Policy,
    states: &mut Vec<Rc<RefCell<State>>>,
    discount_rate: f32,
    threshold: f32,
    iteration_count: Option<usize>,
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
            let new_value: f32 = policy.get_value_of_state(&state.borrow(), discount_rate);

            // Update the state's value and calculate the maximum change (delta)
            state.borrow_mut().set_value(new_value);
            delta = delta.max((v_old - new_value).abs());
        }

        match iteration_count {
            Some(count) => {
                if iteration >= count {
                    println!("Policy Evaluation converged after {} iterations", iteration);
                    break;
                }
            }
            None => {
                if delta < threshold {
                    println!("Policy Evaluation converged after {} iterations", iteration);
                    break;
                }
            }
        }
    }
}

fn poisson_calc(rate_of_occurrence: i64, events_count: u64) -> f64 {
    let factorial: u64 = (1..=events_count).product();
    ((rate_of_occurrence.pow(events_count as u32) as f64) * E.powi((rate_of_occurrence * -1) as i32)) / factorial as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chapter_04::policy::{GreedyPolicy, MutablePolicy, Policy, RandomPolicy};
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

        let mut a1 = Action::new();
        let mut a2 = Action::new();
        let mut a3 = Action::new();
        let mut a4 = Action::new();

        a1.add_possible_next_state(1_f32, s_terminal.clone(), 1_f32);
        a2.add_possible_next_state(0.25, s2.clone(), 0_f32);
        a2.add_possible_next_state(0.75, s3.clone(), 0_f32);
        a3.add_possible_next_state(1_f32, s_terminal.clone(), 14_f32);
        a4.add_possible_next_state(1_f32, s_terminal.clone(), -1_f32);

        s1.borrow_mut().add_action(a1);
        s1.borrow_mut().add_action(a2);
        s2.borrow_mut().add_action(a3);
        s3.borrow_mut().add_action(a4);

        let mut states = vec![s1, s2, s3];

        let simple_policy = RandomPolicy::new();

        println!("before");
        for state in states.iter() {
            println!(
                "state id: {}, value: {}",
                state.borrow().get_id(),
                state.borrow().get_value()
            )
        }

        iterative_policy_evaluation(&simple_policy, &mut states, 0.1, 0.0000001, None);
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

    #[test]
    fn test_grid_world_figure_4_1() {
        let mut states: Vec<Rc<RefCell<State>>> = (0..16)
            .map(|_| Rc::new(RefCell::new(State::new())))
            .collect();
        assert_eq!(states.len(), 16);

        for (id, state) in states.iter().enumerate() {
            if id == 0 || id == 15 {
                continue;
            }
            let row = id / 4;
            let col = id % 4;

            let mut up_action = Action::new();
            let can_move_up = row > 0;
            let up_action_next_state_id = up_action.add_possible_next_state(
                1.0,
                states[if can_move_up { id - 4 } else { id }].clone(),
                -1_f32,
            );

            let mut down_action = Action::new();
            let can_move_down = row < 3;
            down_action.add_possible_next_state(
                1.0,
                states[if can_move_down { id + 4 } else { id }].clone(),
                -1_f32,
            );

            let mut left_action = Action::new();
            let can_move_left = col > 0;
            left_action.add_possible_next_state(
                1.0,
                states[if can_move_left { id - 1 } else { id }].clone(),
                -1_f32,
            );

            let mut right_action = Action::new();
            let can_move_right = col < 3;
            right_action.add_possible_next_state(
                1.0,
                states[if can_move_right { id + 1 } else { id }].clone(),
                -1_f32,
            );

            state.borrow_mut().add_action(up_action);
            state.borrow_mut().add_action(down_action);
            state.borrow_mut().add_action(left_action);
            state.borrow_mut().add_action(right_action);
        }

        let simple_policy = GreedyPolicy::new();

        println!("before");
        for state in states.iter() {
            println!(
                "state id: {}, value: {}",
                state.borrow().get_id(),
                state.borrow().get_value()
            )
        }

        let mut subset = states[1..15].to_vec();

        iterative_policy_evaluation(&simple_policy, &mut subset, 1.0, 0.001, None);
        println!("after");
        for state in states.iter() {
            println!(
                "state id: {}, value: {}",
                state.borrow().get_id(),
                state.borrow().get_value()
            )
        }

        let state_1_values = states[1]
            .borrow()
            .get_debug_value_arr()
            .into_iter()
            .enumerate()
            .map(|(i, v)| -> (f32, f32) { (i as f32, v.clone()) })
            .collect::<Vec<(f32, f32)>>();
        let chart_data_s1 = ChartData::new("State 1".to_string(), state_1_values, BLUE.into());
        let mut chart_builder = ChartBuilder::new();
        chart_builder
            .set_path(PathBuf::from("output/chapter4/gridWorld.png"))
            .set_x_label("Iterations".to_string())
            .set_y_label("Estimated State Value".to_string())
            .set_title("Iterative Policy Evaluation".to_string());
        chart_builder.add_data(chart_data_s1);
        chart_builder.create_chart().unwrap();
    }

    #[test]
    fn is_state_updating_itself() {
        let s1 = Rc::new(RefCell::new(State::new()));
        let s2 = Rc::new(RefCell::new(State::new()));
        let s3 = Rc::new(RefCell::new(State::new()));

        let mut a1 = Action::new();
        let mut a2 = Action::new();

        a1.add_possible_next_state(1_f32, s2.clone(), -1_f32);
        a2.add_possible_next_state(1_f32, s3.clone(), -1_f32);

        s1.borrow_mut().add_action(a1);
        s2.borrow_mut().add_action(a2);

        let mut states = vec![s1, s2];

        let simple_policy = RandomPolicy::new();

        iterative_policy_evaluation(&simple_policy, &mut states, 0.7, 0.01, Some(10));
    }

    #[test]
    fn test_grid_world_convergence_with_policy_iteration() {
        let mut states: Vec<Rc<RefCell<State>>> = (0..16)
            .map(|_| Rc::new(RefCell::new(State::new())))
            .collect();
        assert_eq!(states.len(), 16);

        for (id, state) in states.iter().enumerate() {
            if id == 0 || id == 15 {
                continue;
            }
            let row = id / 4;
            let col = id % 4;

            let mut up_action = Action::new();
            let can_move_up = row > 0;
            let up_action_next_state_id = up_action.add_possible_next_state(
                1.0,
                states[if can_move_up { id - 4 } else { id }].clone(),
                -1_f32,
            );

            let mut down_action = Action::new();
            let can_move_down = row < 3;
            down_action.add_possible_next_state(
                1.0,
                states[if can_move_down { id + 4 } else { id }].clone(),
                -1_f32,
            );

            let mut left_action = Action::new();
            let can_move_left = col > 0;
            left_action.add_possible_next_state(
                1.0,
                states[if can_move_left { id - 1 } else { id }].clone(),
                -1_f32,
            );

            let mut right_action = Action::new();
            let can_move_right = col < 3;
            right_action.add_possible_next_state(
                1.0,
                states[if can_move_right { id + 1 } else { id }].clone(),
                -1_f32,
            );

            state.borrow_mut().add_action(up_action);
            state.borrow_mut().add_action(down_action);
            state.borrow_mut().add_action(left_action);
            state.borrow_mut().add_action(right_action);
        }


        let subset = states[1..15].to_vec();
        let mut simple_policy = MutablePolicy::new(&subset);

        for state in states.iter() {
            println!(
                "state id: {}, value: {}",
                state.borrow().get_id(),
                state.borrow().get_value()
            )
        }

        simple_policy.converge(subset, 1.0, 0.00001);

        for state in states.iter() {
            println!(
                "state id: {}, value: {}",
                state.borrow().get_id(),
                state.borrow().get_value()
            )
        }
    }

    #[test]
    fn test_poisson_calc() {
        let sum: f64 = (0..20).map(|i| {
            poisson_calc(2, i)
        }).sum();

        println!("{}", sum);
    }

    #[test]
    fn test_car_rental_example_4_2() {

    }
}
