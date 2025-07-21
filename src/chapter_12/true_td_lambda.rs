use crate::attempts_at_framework::v1::policy::Policy;
use crate::attempts_at_framework::v2::state::State;
use rand::prelude::IteratorRandom;

pub struct TrueTdLambda<P: Policy, S: State> {
    policy: P,
    size_step_parameter: f64,
    trace_decay_rate: f64,
    discount_rate: f64,
    weights: Vec<f64>,
    starting_states: Vec<S>,
}

impl<P: Policy, S: State> TrueTdLambda<P, S> {
    pub fn new(
        policy: P,
        size_step_parameter: f64,
        trace_decay_rate: f64,
        discount_rate: f64,
        starting_states: Vec<S>,
    ) -> Self {
        if starting_states.is_empty() {
            panic!("Starting states must not be empty");
        }

        let weight_size = starting_states[0].get_values().len();
        let weights = vec![0.0; weight_size];

        Self {
            policy,
            size_step_parameter,
            trace_decay_rate,
            discount_rate,
            weights,
            starting_states,
        }
    }

    pub fn learn_for_single_episode(&mut self) {
        let mut eligibility_trace_vector = vec![0.0; self.weights.len()];
        let mut tmp: f64 = 0.0; // The V_old variable defined as a temporary scalar variable

        let mut current_state = self
            .starting_states
            .iter()
            .choose(&mut rand::rng())
            .unwrap()
            .clone();

        while !current_state.is_terminal() {
            let action = self.chose_action_for_state_according_to_policy(&current_state);
            let (reward, next_state) = current_state.take_action(&action);
            let current_state_value = self.get_state_value(&current_state);
            let next_state_value = self.get_state_value(&next_state);
            let temporal_difference =
                reward + (self.discount_rate * next_state_value) - current_state_value;

            eligibility_trace_vector = eligibility_trace_vector
                .iter()
                .zip(current_state.get_values())
                .map(|(e, v)| {
                    let a = self.discount_rate * self.trace_decay_rate * e;
                    let b = (1.0 - (self.size_step_parameter * a * v)) * v;
                    a + b
                })
                .collect();

            let idk = self.size_step_parameter
                * (temporal_difference + current_state_value - next_state_value);
            let wut = self.size_step_parameter * (current_state_value - tmp);
            self.weights = self
                .weights
                .iter()
                .zip(eligibility_trace_vector.iter())
                .zip(current_state.get_values())
                .map(|((w, e), v)| w + (idk * e) - wut * v)
                .collect();

            tmp = next_state_value;
            current_state = next_state.clone();
        }
    }

    fn chose_action_for_state_according_to_policy(&self, state: &S) -> String {
        match self.policy.select_action_for_state(&state.get_id()) {
            Ok(action) => action,
            Err(_) => state
                .get_actions()
                .iter()
                .choose(&mut rand::rng())
                .unwrap()
                .clone(),
        }
    }

    fn get_state_value(&self, state: &S) -> f64 {
        state
            .get_values()
            .iter()
            .zip(self.weights.iter())
            .map(|(v, w)| v * w)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attempts_at_framework::v1::policy::RandomPolicy;
    use crate::service::x_state_walk_environment::{WalkState, WalkStateFactory};
    fn generate_simple_value_function(total_states: usize) -> impl Fn(WalkState) -> Vec<f64> {
        move |state| {
            if !state.is_terminal() {
                let state_id = state.get_id().parse::<f64>().unwrap();
                let state_feature = state_id / total_states as f64;
                vec![state_feature]
            } else {
                vec![0.0]
            }
        }
    }

    fn generate_polynomial_value_function(
        total_states: usize,
        polynomial_degree: usize,
    ) -> impl Fn(WalkState) -> Vec<f64> {
        move |state| {
            let mut response = vec![0.0; polynomial_degree + 1];
            if !state.is_terminal() {
                response[0] = 1.0;
                for i in 0..polynomial_degree {
                    response[i + 1] = (state.get_id().parse::<f64>().unwrap()
                        / total_states as f64)
                        * response[i];
                }
            }
            response
        }
    }

    #[test]
    fn nineteen_step_random_walk_test() {
        let number_of_episodes = 1000;
        let size_step_parameter = 0.1;
        let trace_decay_rate = 0.9;
        let discount_rate = 1.0;
        let total_states = 19;

        // let value_function = generate_polynomial_value_function(total_states, 1);
        let value_function = generate_simple_value_function(total_states);

        let factory = WalkStateFactory::new(total_states, 1, &value_function).unwrap();

        let starting_state = factory.get_starting_state();
        let starting_states = vec![starting_state];

        let policy = RandomPolicy::new();

        let mut true_td_lambda = TrueTdLambda::new(
            policy,
            size_step_parameter,
            trace_decay_rate,
            discount_rate,
            starting_states,
        );

        (0..number_of_episodes).for_each(|_| true_td_lambda.learn_for_single_episode());

        println!("Weights: {:?}", true_td_lambda.weights);
        (0..total_states).for_each(|i| {
            let (_, state) = factory.generate_state_and_reward_for_id(i as i32);
            let value = true_td_lambda.get_state_value(&state);
            println!("State Id {} has value: {}", i, value);
        })
    }
}
