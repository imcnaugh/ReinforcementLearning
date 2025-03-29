use crate::attempts_at_framework::v1::policy::policy::{Policy, PolicyError};
use crate::attempts_at_framework::v1::policy::StochasticPolicy;

pub struct EGreedyPolicy {
    policy: StochasticPolicy,
    e: f64,
}

impl EGreedyPolicy {
    pub fn new(e: f64) -> Self {
        if e < 0.0 || e > 1.0 {
            panic!("e must be between 0.0 and 1.0");
        }

        Self {
            policy: StochasticPolicy::new(),
            e,
        }
    }

    pub fn set_actions_for_state(
        &mut self,
        state_id: String,
        actions: Vec<String>,
        best_action: String,
    ) {
        if actions.is_empty() {
            panic!("attempting to set state {} to have no actions", state_id);
        }

        if !actions.contains(&best_action) {
            panic!(
                "attempting to set state {} to have best action {} not in actions",
                state_id, best_action
            );
        }

        let x = self.e / actions.len() as f64;
        let best_action_probability = 1.0 - self.e + x;
        let other_action_probability = x;

        let actions_and_odds = actions.into_iter().map(|a| {
            if a == best_action {
                (a, best_action_probability)
            } else {
                (a, other_action_probability)
            }
        });

        self.policy
            .set_actions_for_state(state_id, actions_and_odds.collect());
    }

    pub fn select_greedy_action_for_state(&self, state_id: &str) -> Result<String, Box<PolicyError>> {
        match self.policy.get_actions_for_state(state_id) {
            None => Err(Box::new(PolicyError::new(format!(
                "no actions for state {}",
                state_id
            )))),
            Some(action_and_odds) => {
                let best_action = action_and_odds
                    .iter()
                    .max_by(|x, y| {
                        let x_v = x.1;
                        let y_v = y.1;
                        x_v.partial_cmp(&y_v).unwrap()
                    })
                    .unwrap()
                    .0
                    .clone();
                Ok(best_action)
            }
        }
    }
}

impl Policy for EGreedyPolicy {
    fn select_action_for_state(&self, state_id: &str) -> Result<String, Box<PolicyError>> {
        self.policy.select_action_for_state(state_id)
    }
}
