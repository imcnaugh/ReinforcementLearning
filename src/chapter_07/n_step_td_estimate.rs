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
        let reward = if self.id == 99 { 1.0 } else { 0.0 };
        (reward, BasicState::new(self.id + 1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attempts_at_framework::v1::policy::Policy;
    use std::collections::HashMap;

    fn do_episode<P: Policy>(policy: P) {
        let n = 10;
        let discount_rate: f64 = 0.9;
        let state_values: HashMap<String, f64> = HashMap::new();

        let mut state = BasicState::new(0);
        let mut T = usize::INFINITY;
        let mut t: usize = 0;

        let mut rewards: Vec<f64> = Vec::new();
        loop {
            let state_id = state.get_id();
            if t < T {
                let action = policy.select_action_for_state(&state_id).unwrap();
                let (reward, next_state) = state.take_action(&action);
                rewards.push(reward);
                if next_state.is_terminal() {
                    T = t + 1
                }
            }

            let r = t - n + 1;
            if r >= 0 {
                let g = (r + 1..r + 1 + n)
                    .map(|i| {
                        let pow = i - r - 1;
                        let idk = discount_rate.powi(pow as i32);
                        idk * rewards[i]
                    })
                    .sum::<f64>();

                if r + n < T {
                    let existing_value = state_values.get(&state_id).unwrap_or(&0.0);
                    let value_of_state_at_r_plus_n =
                        existing_value + discount_rate.powi(n as i32) * g;
                    let value = existing_value + discount_rate.powi(n as i32) * g;
                }
            }
            t += 1;
            if r == T - 1 {
                break;
            }
        }
    }

    #[test]
    fn test_n_step_policy_evaluation() {}
}
