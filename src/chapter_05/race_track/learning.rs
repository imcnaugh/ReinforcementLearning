use crate::chapter_05::policy::{DeterministicPolicy, Policy};
use crate::chapter_05::race_track::state::State;
use rand::prelude::*;
use rand::Rng;
use std::collections::HashMap;
use std::rc::Rc;

pub struct MonteCarloOffPolicyLearner<S: State> {
    state_action_values: HashMap<String, f64>,
    state_action_cumulative_rewards: HashMap<String, f64>,
    starting_states: Vec<Rc<S>>,
    states: HashMap<String, Rc<S>>,
    discount_rate: f64,
    target_policy: DeterministicPolicy,
}

impl<S: State> MonteCarloOffPolicyLearner<S> {
    pub fn new(starting_states: Vec<S>, discount_rate: f64) -> Self {
        let starting_states: Vec<Rc<S>> = starting_states
            .into_iter()
            .map(|state| Rc::new(state))
            .collect();
        let states: HashMap<String, Rc<S>> = starting_states
            .iter()
            .map(|state| (state.get_id(), Rc::clone(state)))
            .collect();

        Self {
            state_action_values: HashMap::new(),
            state_action_cumulative_rewards: HashMap::new(),
            target_policy: DeterministicPolicy::new(),
            starting_states,
            discount_rate,
            states,
        }
    }

    pub fn get_target_policy(&self) -> &DeterministicPolicy {
        &self.target_policy
    }

    pub fn learn_for_episodes(&mut self, episode_count: usize) {
        (0..episode_count).for_each(|_| {
            let episode = self.generate_episode();
            let mut g: f64 = 0.0;
            let mut w: f64 = 1.0;
            for (index, (state_id, action, _)) in
                episode[..episode.len() - 1].iter().enumerate().rev()
            {
                g = (self.discount_rate * g) + episode[index + 1].2;
                let state_action_id = format!("{}_{}", state_id, action);
                let new_state_action_cumulative_weight =
                    match self.state_action_cumulative_rewards.get(&state_action_id) {
                        None => w,
                        Some(cumulative_weight) => cumulative_weight + w,
                    };
                self.state_action_cumulative_rewards
                    .insert(state_action_id.clone(), new_state_action_cumulative_weight);
                let current_state_action_value = self
                    .state_action_values
                    .get(&state_action_id)
                    .unwrap_or(&0.0);
                let new_state_action_value = current_state_action_value
                    + ((w / new_state_action_cumulative_weight) * (g - current_state_action_value));
                self.state_action_values
                    .insert(state_action_id.clone(), new_state_action_value);

                let (_, best_action) = self
                    .states
                    .get(state_id)
                    .unwrap()
                    .get_actions()
                    .iter()
                    .fold(
                        (f64::MIN, String::new()),
                        |(best_action_value, best_action), action| {
                            let action_id = format!("{}_{}", state_id, action);
                            let action_value = self
                                .state_action_values
                                .get(&action_id)
                                .unwrap_or(&f64::MIN);
                            if *action_value > best_action_value {
                                (*action_value, action.clone())
                            } else {
                                (best_action_value, best_action)
                            }
                        },
                    );
                self.target_policy
                    .set_action_for_state(state_id, &best_action);

                if best_action != *action {
                    break;
                }

                w = w * (1.0 / self.states.get(state_id).unwrap().get_actions().len() as f64);
            }
        })
    }

    fn generate_episode(&mut self) -> Vec<(String, String, f64)> {
        let mut rng = rand::rng();
        let mut current_state = Rc::clone(self.starting_states.choose(&mut rng).unwrap());
        let mut history: Vec<(String, String, f64)> = Vec::new();
        while !current_state.is_terminal() {
            let current_state_id = current_state.get_id();
            let actions = current_state.get_actions();
            let action = actions.choose(&mut rng).unwrap();
            let (reward, new_state) = current_state.take_action(action);
            history.push((current_state_id.clone(), action.clone(), reward));
            let next_state = Rc::new(new_state);
            self.states
                .insert(next_state.get_id().clone(), Rc::clone(&next_state));
            current_state = next_state;
        }
        history
    }
}

#[cfg(test)]
mod tests {
    use super::*;
}
