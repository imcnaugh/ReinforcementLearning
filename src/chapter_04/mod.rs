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

/// # Iterative Policy Evaluation
///
/// Given a set of states with actions that connect each other, This function iterates over the
/// states until all states value converges. Since a states value is calculated by multiplying the
/// next states value by the odds of that next state being chosen, and since all states are given
/// an arbitrary starting value. we need to loop over each state calculating out its value until
/// it starts to converge.
///
/// ## E.G.
///
/// if we have 3 states,
///  - A, has a single action that leads to state B 100% of the time with no reward
///  - B, has a single action that leads to state C 100% of the time with a reward of 10
///  - C, a terminal state
///
/// Let's assign an arbitrary value of 0 to all states. and a discount_rate of 1
///
/// In the first iteration, the new value of A will be calculated off the value of state B, with its
/// initial value of 0 the calculation would be 0 + (0 * 1) (the reward of the action + the value of
/// b times the odds of that state being selected).
///
/// Then state B would be calculated, with its single action with a reward of 10 leading to state C
/// the math looks like (10 + (0 * 1))  leaving the value of B as 10.
///
/// C is a terminal state, so it would not be updated.
///
/// On the second loop the value of A would be updated with the new value of B, so the math would
/// look like (0 + (10 * 1)) again, the reward plus the value of the new state times the odds of it
/// being chosen.
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
            let new_value: f32 = policy.calc_state_value(&state.borrow(), discount_rate);

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

pub fn poisson_calc(rate_of_occurrence: i64, events_count: u64) -> f64 {
    let factorial: u64 = (1..=events_count).product();
    ((rate_of_occurrence.pow(events_count as u32) as f64)
        * E.powi((rate_of_occurrence * -1) as i32))
        / factorial as f64
}

pub fn value_iteration(states: &mut Vec<Rc<RefCell<State>>>, discount_rate: f32, threshold: f32) {
    let mut iteration = 0;
    loop {
        iteration += 1;
        let mut delta: f32 = 0.0;
        for mut state in states.iter_mut() {
            if state.borrow().get_is_terminal() {
                continue;
            }

            let old_value = state.borrow().get_value();
            let mut new_value = f32::NEG_INFINITY;
            state.borrow().get_actions().iter().for_each(|action| {
                let action_value = action.get_value(discount_rate);
                if new_value < action_value {
                    new_value = action_value;
                }
            });

            state.borrow_mut().set_value(new_value);
            delta = delta.max((old_value - new_value).abs());
        }

        if delta < threshold {
            println!("Value Iteration converged after {} iterations", iteration);
            break;
        }
    }
    println!("finished estimating state value for policy");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chapter_04::policy::{GreedyPolicy, MutablePolicy, Policy, RandomPolicy};
    use crate::service::{LineChartBuilder, LineChartData};
    use plotters::prelude::{BLUE, GREEN, RED};
    use std::cell::RefCell;
    use std::collections::HashMap;
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

        let mut chart_builder = LineChartBuilder::new();
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
            let data =
                LineChartData::new(format!("State {}", next_id), points, styles.next().unwrap());
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
        let chart_data_s1 = LineChartData::new("State 1".to_string(), state_1_values, BLUE.into());
        let mut chart_builder = LineChartBuilder::new();
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

        simple_policy.policy_iteration(subset, 1.0, 0.00001);

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
        (0..20).for_each(|i| {
            println!("prob of {i}: {}", poisson_calc(2, i));
        });

        let sum: f64 = (0..20).map(|i| poisson_calc(2, i)).sum();

        println!("{}", sum);

        println!("---------------");
        let odds_a = (0..20).map(|i| poisson_calc(2, i)).collect::<Vec<f64>>();
        let odds_b = (0..20).map(|i| poisson_calc(3, i)).collect::<Vec<f64>>();

        let sum = odds_a
            .iter()
            .map(|a| odds_b.iter().map(|b| a * b).sum::<f64>())
            .sum::<f64>();

        println!("sum of both {}", sum);
    }

    #[test]
    fn range_test() {
        (-5..=5).for_each(|i| {
            println!("{}", i);
        });
    }

    #[test]
    fn max_tests() {
        let idk = 20.min(40);
        println!("{}", idk);
    }

    #[test]
    fn test_car_rental_example_4_2() {
        let max_cars: i32 = 10;
        let max_cars_moved_per_night: usize = 5;

        let state_count = (max_cars + 1) * (max_cars + 1);
        let mut states = (0..state_count)
            .map(|i| {
                let mut state = State::new();
                state.add_action(Action::new());
                Rc::new(RefCell::new(state))
            })
            .collect::<Vec<Rc<RefCell<State>>>>();

        let location_1_rental_odds = (0..=max_cars)
            .map(|i| poisson_calc(3, i as u64))
            .collect::<Vec<f64>>();
        let location_2_rental_odds = (0..=max_cars)
            .map(|i| poisson_calc(4, i as u64))
            .collect::<Vec<f64>>();
        let location_1_return_odds = (0..=max_cars)
            .map(|i| poisson_calc(3, i as u64))
            .collect::<Vec<f64>>();
        let location_2_return_odds = (0..=max_cars)
            .map(|i| poisson_calc(2, i as u64))
            .collect::<Vec<f64>>();

        states.iter().enumerate().for_each(|(index, state)| {
            let cars_at_first_location = index / (max_cars as usize + 1);
            let cars_at_second_location = index % (max_cars as usize + 1);

            println!(
                "cars at first location: {}, cars at second location: {}, index: {}",
                cars_at_first_location, cars_at_second_location, index
            );

            let mut actions: Vec<Action> = vec![];

            for i in (max_cars_moved_per_night as i32 * -1)..=max_cars_moved_per_night as i32 {
                let mut new_action = Action::new();
                new_action.set_description(format!("{}", i));

                let cars_moved_to_l1: i32 = i * -1;
                let cars_moved_to_l2: i32 = i;

                for (l_1_rental_requests, l1rent_odds) in location_1_rental_odds.iter().enumerate()
                {
                    for (l_2_rental_requests, l2rent_odds) in
                        location_2_rental_odds.iter().enumerate()
                    {
                        for (l_1_returned_cars, l1return_odds) in
                            location_1_return_odds.iter().enumerate()
                        {
                            for (l_2_returned_cars, l2return_odds) in
                                location_2_return_odds.iter().enumerate()
                            {
                                let odds =
                                    (l1rent_odds * l2rent_odds * l1return_odds * l2return_odds)
                                        as f32;

                                let cars_at_l_1 = cars_at_first_location as i32 + cars_moved_to_l1;
                                let cars_at_l_2 = cars_at_second_location as i32 + cars_moved_to_l2;

                                if cars_at_l_1 < 0 || cars_at_l_2 < 0 {
                                    continue;
                                }

                                let cars_rented_at_l_1 =
                                    l_1_rental_requests.min(cars_at_l_1 as usize);
                                let cars_remaining_at_l_1 = max_cars.min(
                                    cars_at_l_1 - cars_rented_at_l_1 as i32
                                        + l_1_returned_cars as i32,
                                );

                                let cars_rented_at_l_2 =
                                    l_2_rental_requests.min(cars_at_l_2 as usize);
                                let cars_remaining_at_l_2 = max_cars.min(
                                    cars_at_l_2 - cars_rented_at_l_2 as i32
                                        + l_2_returned_cars as i32,
                                );

                                let reward: f32 = ((cars_rented_at_l_1 * 10) as i32
                                    + (cars_rented_at_l_2 * 10) as i32
                                    - (i.abs() * 2))
                                    as f32;
                                let next_state = states[(cars_remaining_at_l_1 * (max_cars + 1))
                                    as usize
                                    + cars_remaining_at_l_2 as usize]
                                    .clone();
                                new_action.add_possible_next_state(odds, next_state, reward);
                            }
                        }
                    }
                }

                actions.push(new_action);
            }

            actions.into_iter().for_each(move |a| {
                state.borrow_mut().add_action(a);
            });
        });

        let mut policy = MutablePolicy::new(&states);

        let subset = states[..].to_vec();

        policy.policy_iteration(subset, 0.9, 0.001);

        states
            .chunks((max_cars + 1) as usize)
            .enumerate()
            .for_each(|(x, chunk)| {
                let cars_at_first_location = x % (max_cars + 1) as usize;

                println!(
                    "l1 cars: {}: \t{}",
                    cars_at_first_location,
                    chunk
                        .iter()
                        .map(|state| policy
                            .get_optimal_action_for_state(&state.borrow())
                            .unwrap()
                            .get_description()
                            .unwrap_or(&"".to_string())
                            .clone())
                        // .map(|state| format!("{:.2}",state.borrow().get_value()))
                        .collect::<Vec<String>>()
                        .join("\t")
                );
            });

        println!();
        println!("0 .. 20");
        println!("l2");
    }

    #[test]
    fn test_gamblers_problem_example_4_3() {
        let winning_odds = 0.4;
        let winning_amount = 100;
        let losing_amount = 0;
        let lowest_bet = losing_amount + 1;
        let losing_odds = 1_f32 - winning_odds;

        println!("Setting up states");
        let mut states = (lowest_bet..winning_amount)
            .map(|i| {
                let mut new_state = State::new();
                new_state.set_id(format!("{}", i));
                new_state.set_capital(i);
                Rc::new(RefCell::new(new_state))
            })
            .collect::<Vec<Rc<RefCell<State>>>>();
        let terminal_state = Rc::new(RefCell::new(State::new()));
        terminal_state
            .borrow_mut()
            .set_id("Terminal State".to_string());
        terminal_state.borrow_mut().set_is_terminal(true);
        states.push(terminal_state.clone());
        println!("States setup complete");

        println!("Setting up actions");
        states.iter().enumerate().for_each(|(index, state)| {
            if state.borrow().get_is_terminal() {
                return;
            }

            let capital = state.borrow().get_capital().unwrap().clone();

            let mut actions: Vec<Action> = vec![];
            (losing_amount..=capital).for_each(|i| {
                let mut bet_action = Action::new();
                bet_action.set_description(format!("{}", i));
                let total_after_win = capital + i;
                let total_after_loss = capital - i;
                let win_reward = if total_after_win >= winning_amount {
                    1_f32
                } else {
                    0_f32
                };
                let lose_reward = if total_after_loss <= losing_amount {
                    0_f32
                } else {
                    0_f32
                };

                let winning_next_state = if total_after_win >= winning_amount {
                    &terminal_state
                } else {
                    &states[total_after_win as usize]
                };
                let losing_next_state = if total_after_loss <= losing_amount {
                    &terminal_state
                } else {
                    &states[total_after_loss as usize]
                };

                bet_action.add_possible_next_state(
                    winning_odds,
                    winning_next_state.clone(),
                    win_reward,
                );
                bet_action.add_possible_next_state(
                    losing_odds,
                    losing_next_state.clone(),
                    lose_reward,
                );
                actions.push(bet_action);
            });

            actions.into_iter().for_each(move |a| {
                state.borrow_mut().add_action(a);
            });
        });
        println!("Actions setup complete");

        states.iter().for_each(|state| {
            println!(
                "state capitol: {}, value: {}",
                state.borrow().get_capital().unwrap_or(0),
                state.borrow().get_value()
            );
        });

        println!("Setting up policy");
        value_iteration(&mut states, 1.0, 0.00001);
        println!("Policy convergence complete");

        states.iter().for_each(|state| {
            println!(
                "state capitol: {}, value: {}",
                state.borrow().get_capital().unwrap_or(0),
                state.borrow().get_value()
            );
        });

        println!("Graphing output");
        let mut optimal_bet_per_capital: Vec<(f32, f32)> = vec![];
        states[..].iter().enumerate().for_each(|(index, state)| {
            let action = state.borrow().get_max_action_description(1.0);
            let best_bet_amount = action.parse::<i32>().unwrap_or(0) as f32;
            let current_capital = index as f32 + 1.0;
            optimal_bet_per_capital.push((current_capital, best_bet_amount));
        });
        let state_0_values = states[8]
            .borrow()
            .get_debug_value_arr()
            .iter()
            .enumerate()
            .map(|(i, v)| (i as f32, v.clone()))
            .collect::<Vec<(f32, f32)>>();

        println!("printing graph");
        let best_bet_at_capitol_data = LineChartData::new(
            "Best Bets".to_string(),
            optimal_bet_per_capital,
            BLUE.into(),
        );
        let state_0_data = LineChartData::new("State 0".to_string(), state_0_values, RED.into());
        let mut chart_builder = LineChartBuilder::new();
        chart_builder
            .set_path(PathBuf::from("output/chapter4/Gambler.png"))
            .set_x_label("Capital".to_string())
            .set_y_label("Final Policy".to_string())
            .set_title("Gamblers Problem Example 4.3".to_string());
        chart_builder.add_data(best_bet_at_capitol_data);
        chart_builder.add_data(state_0_data);
        chart_builder.create_chart().unwrap();
    }
}
