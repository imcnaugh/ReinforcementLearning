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
        let mut current_state = starting_state;
        let mut action = self.select_action(&current_state);

        while !current_state.is_terminal() {}
    }

    fn select_action<S: State>(&self, state: &S) -> String {
        let mut rng = rand::rng();
        if rng.random::<f64>() < self.explore_rate {
            let actions = state.get_actions();
            return actions.choose(&mut rng).unwrap().clone();
        }

        self.get_best_action_for_state(state)
    }

    fn get_best_action_for_state<S: State>(&self, state: &S) -> String {
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
}
