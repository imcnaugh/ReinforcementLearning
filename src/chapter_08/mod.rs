use crate::attempts_at_framework::v1::policy::{EGreedyPolicy, Policy};
use crate::attempts_at_framework::v1::state::State;
use rand::prelude::IteratorRandom;
use std::collections::HashMap;

pub fn tabular_dyna_q<S: State>(
    iteration_count: usize,
    states: Vec<S>,
    discount_rate: f64,
    size_step_parameter: f64,
    n: usize,
) -> EGreedyPolicy {
    let state_map: HashMap<String, S> = states.iter().map(|s| (s.get_id(), s.clone())).collect();
    let mut state_action_values: HashMap<String, f64> = HashMap::new();
    let mut model: HashMap<(String, String), (f64, String)> = HashMap::new();
    let mut policy: EGreedyPolicy = EGreedyPolicy::new(0.1);
    let mut rng = rand::rng();
    (0..iteration_count).for_each(|episode_count| {
        let mut state = state_map.get("0_2").unwrap().clone();

        while !state.is_terminal() {
            let action = match policy.select_action_for_state(&state.get_id()) {
                Ok(a) => a,
                Err(_) => state.get_actions().iter().choose(&mut rng).unwrap().clone(),
            };

            let state_action_id = get_state_action_id(&state.get_id(), &action);
            let (reward, next_state) = state.take_action(&action);
            let current_state_action_value =
                state_action_values.get(&state_action_id).unwrap_or(&0.0);
            let next_state_best_action_value =
                get_max_value_of_state_actions(&state_action_values, &next_state)
                    .unwrap_or((0.0, String::new()));
            let new_state_action_value = current_state_action_value
                + (size_step_parameter
                    * (reward + (discount_rate * next_state_best_action_value.0)
                        - current_state_action_value));
            state_action_values.insert(state_action_id.clone(), new_state_action_value);
            model.insert(
                (state.get_id(), action),
                (new_state_action_value, next_state.get_id()),
            );

            if episode_count != 0 {
                (0..n).for_each(|_| {
                    let ((s, a), (r, ns)) = model.iter().choose(&mut rng).unwrap();
                    let s_a_id = get_state_action_id(s, a);
                    let current_s_a_value = state_action_values.get(&s_a_id).unwrap_or(&0.0);
                    let ns = match state_map.get(ns) {
                        None => panic!("state {} not found", ns),
                        Some(s) => s.clone(),
                    };
                    let (best_ns_value, best_ns_action) =
                        get_max_value_of_state_actions(&state_action_values, &ns)
                            .unwrap_or((0.0, String::new()));
                    let new_s_a_value = current_s_a_value
                        + (size_step_parameter
                            * (r + (discount_rate * best_ns_value) - current_s_a_value));
                    state_action_values.insert(s_a_id.clone(), new_s_a_value);
                    if best_ns_action != "" {
                        policy.set_actions_for_state(
                            ns.get_id(),
                            ns.get_actions().clone(),
                            best_ns_action.clone(),
                        );
                    }
                });
            }

            state = next_state;
        }
    });

    policy
}

fn get_max_value_of_state_actions<S: State>(
    state_action_values: &HashMap<String, f64>,
    state: &S,
) -> Option<(f64, String)> {
    let actions = state.get_actions();
    let mut max_value: Option<(f64, String)> = None;
    for action in actions {
        let state_action_id = get_state_action_id(&state.get_id(), &action);
        if let Some(value) = state_action_values.get(&state_action_id) {
            match &max_value {
                None => max_value = Some((value.clone(), action.clone())),
                Some(current_max) => {
                    if value > &current_max.0 {
                        max_value = Some((value.clone(), action.clone()));
                    }
                }
            }
        }
    }
    max_value
}

fn get_state_action_id(state_id: &str, action_id: &str) -> String {
    format!("{}_{}", state_id, action_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn figure_8_2() {
        let wall_ids = vec!["1_2", "2_2", "3_2", "4_5", "0_7", "1_7", "2_7"];
        let wall_ids: Vec<String> = wall_ids.iter().map(|s| s.to_string()).collect();

        let states: Vec<TestState> = (0..6)
            .flat_map(|row| (0..9).map(move |col| TestState::new(row, col)))
            .filter(|s| !wall_ids.contains(&s.id))
            .collect();

        let policy = tabular_dyna_q(10, states, 0.5, 0.1, 5);
        let determ = policy.to_deterministic_policy();

        (0..6).for_each(|row| {
            (0..9).for_each(|col| {
                let id = format!("{}_{}", row, col);
                if wall_ids.contains(&id) {
                    print!("#");
                    return;
                }
                let action = determ
                    .select_action_for_state(&id)
                    .unwrap_or(String::from(" "));
                let char = match action.as_str() {
                    "left" => "<",
                    "right" => ">",
                    "up" => "^",
                    "down" => "v",
                    _ => " ",
                };
                print!("{}", char);
            });
            println!();
        });
    }

    #[derive(Clone)]
    struct TestState {
        id: String,
        row: usize,
        col: usize,
    }

    impl TestState {
        fn new(row: usize, col: usize) -> Self {
            let id = format!("{}_{}", row, col);
            Self { id, row, col }
        }
    }

    impl State for TestState {
        fn get_id(&self) -> String {
            self.id.clone()
        }

        fn get_actions(&self) -> Vec<String> {
            let mut can_move_up = self.row > 0;
            let mut can_move_down = self.row < 5;
            let mut can_move_left = self.col > 0;
            let mut can_move_right = self.col < 8;

            if (1..=3).contains(&self.row) {
                if self.col == 1 {
                    can_move_right = false;
                }
                if self.col == 3 {
                    can_move_left = false;
                }
            }

            if self.col == 2 {
                if self.row == 0 {
                    can_move_down = false;
                }
                if self.row == 4 {
                    can_move_up = false;
                }
            }

            if self.col == 5 {
                if self.row == 3 {
                    can_move_down = false;
                }
                if self.row == 5 {
                    can_move_up = false;
                }
            }

            if self.row == 4 {
                if self.col == 4 {
                    can_move_right = false;
                }
                if self.col == 6 {
                    can_move_left = false;
                }
            }

            if (0..=2).contains(&self.row) {
                if self.col == 6 {
                    can_move_right = false;
                }
                if self.col == 8 {
                    can_move_left = false;
                }
            }

            if self.col == 7 && self.row == 3 {
                can_move_up = false;
            }

            let mut actions = Vec::new();
            if can_move_up {
                actions.push("up".to_string());
            }
            if can_move_down {
                actions.push("down".to_string());
            }
            if can_move_left {
                actions.push("left".to_string());
            }
            if can_move_right {
                actions.push("right".to_string());
            }

            actions
        }

        fn is_terminal(&self) -> bool {
            self.col == 8 && self.row == 0
        }

        fn take_action(&self, action: &str) -> (f64, Self) {
            let (new_row, new_col) = match action {
                "up" => (self.row - 1, self.col),
                "down" => (self.row + 1, self.col),
                "left" => (self.row, self.col - 1),
                "right" => (self.row, self.col + 1),
                _ => panic!("Invalid action"),
            };

            let reward = if new_row == 0 && new_col == 8 {
                1.0
            } else {
                0.0
            };

            let new_state = TestState::new(new_row, new_col);
            (reward, new_state)
        }
    }
}
