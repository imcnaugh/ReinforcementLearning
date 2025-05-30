use crate::attempts_at_framework::v2::artificial_neural_network::model::Model;
use crate::attempts_at_framework::v2::state::State;
use rand::Rng;
use std::collections::VecDeque;

pub struct NStepTD {
    n: usize,
    discount_rate: f64,
    learning_rate: f64,
    explore_rate: f64,
    episodes_learned_for: usize,
    model: Model,
}

impl NStepTD {
    pub fn new(n: usize, model: Model, learning_rate: f64) -> Self {
        Self {
            n,
            model,
            learning_rate,
            explore_rate: 0.1,
            discount_rate: 1.0,
            episodes_learned_for: 0,
        }
    }

    pub fn learn_from_episode<S: State>(&mut self, starting_state: S) {
        let mut current_state = starting_state;
        let mut states_queue: VecDeque<S> = VecDeque::new();
        let mut rewards_queue: VecDeque<f64> = VecDeque::new();

        while !current_state.is_terminal() {
            let action: String = self.select_next_action(&current_state);
            let (reward, next_state) = current_state.take_action(&action);
            states_queue.push_back(current_state);
            rewards_queue.push_back(reward);

            if states_queue.len() >= self.n {
                let old_state = states_queue.pop_front().unwrap();
                let mut n_step_return = 0.0;

                for (i, r) in rewards_queue.iter().enumerate() {
                    n_step_return += r * self.discount_rate.powi(i as i32);
                }

                if !next_state.is_terminal() {
                    n_step_return += self.discount_rate.powi(self.n as i32)
                        * self.model.predict(next_state.get_values())[0];
                }

                self.model.train(
                    old_state.get_values(),
                    vec![n_step_return],
                    self.learning_rate,
                );

                rewards_queue.pop_front();
            }

            current_state = next_state;
        }

        while !states_queue.is_empty() {
            let old_state = states_queue.pop_front().unwrap();
            let mut n_step_return = 0.0;

            for (i, r) in rewards_queue.iter().enumerate() {
                n_step_return += r * self.discount_rate.powi(i as i32);
            }

            self.model.train(
                old_state.get_values(),
                vec![n_step_return],
                self.learning_rate,
            );

            rewards_queue.pop_front();
        }
    }

    /// TODO fix this, this is the real problem
    /// How do i estimate the value of a position, i guess that's the
    /// real question im trying to answer. If my training set contains
    /// only states where it is my turn to move, will that same model
    /// work on states where it is not my turn. Probably not, at least
    /// not without adding them to the training set, but with the current
    /// structure that's not possible. So V1 of ANN then? ugh.
    ///
    /// But this is a larger problem, if I can train this up, how do i use
    /// the model to chose moves going forward. I could use a heuristic
    /// search of depth 2 to get back to states where its my move. but
    /// that feels bad when I type it out.
    ///
    /// Ugh according to DeepSeek its best to use a Policy Gradient method
    /// for this type of task, that's covered in chapter 13. So time to get
    /// back to reading, but ill be back!c
    fn select_next_action<S: State>(&self, state: &S) -> String {
        if rand::rng().random::<f64>() < self.explore_rate {
            let actions = state.get_actions();
            let random_index = rand::rng().random_range(0..actions.len());
            return actions[random_index].clone();
        }
        self.select_best_action_for_state(state)
    }

    pub fn select_best_action_for_state<S: State>(&self, state: &S) -> String {
        let actions = state.get_actions();
        let mut best_value = f64::MIN;
        let mut best_action = actions.first().unwrap().clone();

        for action in actions {
            let state_clone = state.clone();
            let (reward, next_state) = state_clone.take_action(&action);
            let value = self.model.predict(next_state.get_values())[0];
            if (value + reward) > best_value {
                best_action = action.clone();
                best_value = value;
            }
        }

        best_action
    }

    pub fn set_discount_rate(&mut self, discount_rate: f64) {
        self.discount_rate = discount_rate;
    }

    pub fn set_learning_rate(&mut self, learning_rate: f64) {
        self.learning_rate = learning_rate;
    }

    pub fn set_explore_rate(&mut self, explore_rate: f64) {
        self.explore_rate = explore_rate;
    }

    pub fn get_discount_rate(&self) -> f64 {
        self.discount_rate
    }

    pub fn get_learning_rate(&self) -> f64 {
        self.learning_rate
    }

    pub fn get_explore_rate(&self) -> f64 {
        self.explore_rate
    }

    pub fn get_episodes_learned_for(&self) -> usize {
        self.episodes_learned_for
    }

    pub fn get_model(&self) -> &Model {
        &self.model
    }
}
