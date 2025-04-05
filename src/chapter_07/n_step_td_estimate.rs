use crate::attempts_at_framework::v1::state::State;

#[derive(Debug, Clone)]
pub struct BasicState {
    id: usize,
}

impl BasicState {
    pub fn new(id: usize) -> Self {
        Self { id }
    }
}

impl State for BasicState {
    fn get_id(&self) -> String {
        self.id.to_string()
    }

    fn get_actions(&self) -> Vec<String> {
        vec![String::from("right")]
    }

    fn is_terminal(&self) -> bool {
        self.id == 100
    }

    fn take_action(&self, action: &str) -> (f64, Self) {
        assert_eq!(action, "right");
        // let reward = if self.id % 10 == 0 { 1.0 } else { 0.0 };
        let reward = if self.id == 99 { 1.0 } else { 0.0 };
        (reward, BasicState::new(self.id + 1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attempts_at_framework::v1::policy::{DeterministicPolicy, Policy};
    use std::collections::HashMap;

    fn do_episode<P: Policy>(policy: P) {
        let n = 10;
        let discount_rate: f64 = 1.0;
        let size_step_parameter: f64 = 0.1;
        let mut state_values: HashMap<String, f64> = HashMap::new();

        let mut state = BasicState::new(0);
        let mut next_state: Option<BasicState> = None;
        let mut T = i32::MAX;
        let mut t: i32 = 0;

        let mut rewards: Vec<f64> = vec![0.0; 101];
        loop {
            let state_id = state.get_id();
            if t < T {
                let action = policy.select_action_for_state(&state_id).unwrap();
                let (reward, ns) = state.take_action(&action);
                rewards[t as usize] = reward;
                if ns.is_terminal() {
                    T = t + 1
                }
                next_state = Some(ns);
            }

            let r = t - n + 1;
            if r >= 0 {
                let mut g = (r + 1..=(r + 1 + n).min(T))
                    .map(|i| {
                        let pow = i - r - 1;
                        let idk = discount_rate.powi(pow);
                        idk * rewards[i as usize]
                    })
                    .sum::<f64>();

                if r + n < T {
                    let value_of_state_at_r_plus_n =
                        state_values.get(&(n + r).to_string()).unwrap_or(&0.0);
                    g = (value_of_state_at_r_plus_n * discount_rate.powi(n)) + g;
                }
                let existing_value = state_values.get(&state_id).unwrap_or(&0.0);
                let new_value = existing_value + (size_step_parameter * (g - existing_value));
                state_values.insert(state_id, new_value);
            }
            t += 1;
            state = next_state.clone().unwrap();
            if r == T - 1 {
                break;
            }
        }

        (90..100).for_each(|s| {
            println!("{}: {}", s, state_values.get(&s.to_string()).unwrap());
        })
    }

    #[test]
    fn test_n_step_policy_evaluation() {
        let mut policy = DeterministicPolicy::new();
        (0..=99).for_each(|s| policy.set_actions_for_state(s.to_string(), String::from("right")));

        do_episode(policy);
    }
}
