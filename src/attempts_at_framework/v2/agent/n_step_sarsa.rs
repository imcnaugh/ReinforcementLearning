use crate::attempts_at_framework::v2::artificial_neural_network::model::Model;
use crate::attempts_at_framework::v2::state::State;
use rand::prelude::IndexedRandom;
use rand::Rng;

pub struct NStepSarsa {
    n: usize,
    discount_rate: f64,
    learning_rate: f64,
    explore_rate: f64,
    episodes_learned_for: usize,
    model: Model,
}

impl NStepSarsa {
    pub fn new(
        n: usize,
        discount_rate: f64,
        learning_rate: f64,
        explore_rate: f64,
        model: Model,
    ) -> Self {
        Self {
            n,
            discount_rate,
            learning_rate,
            explore_rate,
            episodes_learned_for: 0,
            model,
        }
    }

    pub fn learn_from_episode<S: State>(&mut self, starting_state: S) {
        let mut terminal_time: Option<usize> = None;
        let mut current_state = starting_state;
        let mut current_action = self.select_action(&current_state);
        let mut next_state: Option<S> = None;
        let mut next_action: Option<String> = None;
        let mut states_actions_and_rewards: Vec<(S, String, f64)> = Vec::new();
        states_actions_and_rewards.push((current_state.clone(), current_action.clone(), 0.0));

        for time_step in 0..usize::MAX {
            if time_step < terminal_time.unwrap_or(usize::MAX) {
                let (reward, ns) = current_state.take_action(&current_action);
                if ns.is_terminal() {
                    terminal_time = Some(time_step + 1);
                } else {
                    next_action = Some(self.select_action(&ns));
                    states_actions_and_rewards.push((
                        ns.clone(),
                        next_action.clone().unwrap(),
                        reward,
                    ));
                    next_state = Some(ns);
                }
            }

            let time_step_to_update = time_step as i32 - self.n as i32 + 1;
            if time_step_to_update >= 0 {
                let f = (time_step_to_update as usize + 1).min(terminal_time.unwrap_or(usize::MAX));
                let l = (time_step_to_update as usize + self.n)
                    .min(terminal_time.unwrap_or(usize::MAX));
                let sum_of_rewards = &states_actions_and_rewards[f..l]
                    .iter()
                    .enumerate()
                    .map(|(index, (_, _, r))| {
                        let pow = (self.n - index) as i32;
                        let idk = self.discount_rate.powi(pow);
                        idk * *r
                    })
                    .sum::<f64>();

                let state_value_at_r_plus_n = if time_step_to_update as usize + self.n
                    < terminal_time.unwrap_or(usize::MAX)
                {
                    let adjusted_discount_rate = self.discount_rate.powi(self.n as i32);
                    let (s, a, _) =
                        &states_actions_and_rewards[time_step_to_update as usize + self.n];
                    let s_values = self.adjust_values(s, a.clone());
                    let state_value = self.model.predict(s_values)[0];
                    state_value * adjusted_discount_rate
                } else {
                    0.0
                };

                let total_reward = sum_of_rewards + state_value_at_r_plus_n;
                let (state_to_update, action_chosen, _) =
                    &states_actions_and_rewards[(time_step_to_update - 1).max(0) as usize];
                let adjusted_values_of_state =
                    self.adjust_values(state_to_update, action_chosen.clone());
                self.model.train(
                    adjusted_values_of_state,
                    vec![total_reward],
                    self.learning_rate,
                );
            }

            let terminal_time_as_i32 = match terminal_time {
                None => i32::MAX,
                Some(t) => t as i32,
            };

            if time_step_to_update == terminal_time_as_i32 {
                break;
            }

            current_state = next_state.unwrap_or(current_state);
            next_state = None;
            current_action = next_action.unwrap_or(current_action);
            next_action = None;
        }

        self.episodes_learned_for += 1;
    }

    fn select_action<S: State>(&self, state: &S) -> String {
        let mut rng = rand::rng();
        if rng.random::<f64>() < self.explore_rate {
            let actions = state.get_actions();
            return actions.choose(&mut rng).unwrap().clone();
        }

        self.get_best_action_for_state(state)
    }

    pub fn get_best_action_for_state<S: State>(&self, state: &S) -> String {
        let actions: Vec<(String, f64)> = state
            .get_actions()
            .iter()
            .map(|action| {
                let adjusted_values = self.adjust_values(state, action.clone());
                let estimated_value = self.model.predict(adjusted_values)[0];
                (action.clone(), estimated_value)
            })
            .collect();

        let mut best_action = actions[0].clone();
        for action in actions {
            if action.1 > best_action.1 {
                best_action = action;
            }
        }
        best_action.0
    }

    fn adjust_values<S: State>(&self, state: &S, action: String) -> Vec<f64> {
        let values = state.get_values();
        let action_count = state.get_actions().len();
        let action_index = state
            .get_actions()
            .iter()
            .position(|a| a == &action)
            .unwrap();

        let mut result = vec![0.0; values.len() * action_count];
        let start_index = action_index * values.len();
        result[start_index..start_index + values.len()].copy_from_slice(&values);
        result
    }

    pub fn print_weights(&self) {
        self.model.print_weights();
    }
}
