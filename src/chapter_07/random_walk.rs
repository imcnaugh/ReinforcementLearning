use crate::attempts_at_framework::v1::policy::{Policy, RandomPolicy};
use crate::attempts_at_framework::v1::state::State;
use std::collections::HashMap;

#[derive(Clone)]
pub struct RandomWalkNode<'a> {
    id: usize,
    environment: &'a RandomWalkEnvironment,
}

pub struct RandomWalkEnvironment {
    num_of_nodes: usize,
    start_node: usize,
    left_reward: f64,
    right_reward: f64,
}

impl<'a> RandomWalkNode<'a> {
    pub fn new(environment: &'a RandomWalkEnvironment, id: usize) -> Self {
        Self { id, environment }
    }
}

impl State for RandomWalkNode<'_> {
    fn get_id(&self) -> String {
        self.id.to_string()
    }

    fn get_actions(&self) -> Vec<String> {
        let mut actions = vec![];
        if self.id > 0 {
            actions.push(String::from("left"));
        }
        if self.id < self.environment.num_of_nodes - 1 {
            actions.push(String::from("right"));
        }
        actions
    }

    fn is_terminal(&self) -> bool {
        if self.environment.num_of_nodes == self.id {
            true
        } else if self.id == 0 {
            true
        } else {
            false
        }
    }

    fn take_action(&self, action: &str) -> (f64, Self) {
        match action {
            "left" => {
                let reward = if self.id - 1 == 0 {
                    self.environment.left_reward
                } else {
                    0.0
                };
                (reward, RandomWalkNode::new(self.environment, self.id - 1))
            }
            "right" => {
                let reward = if self.id + 1 == self.environment.num_of_nodes {
                    self.environment.right_reward
                } else {
                    0.0
                };
                (reward, RandomWalkNode::new(self.environment, self.id + 1))
            }
            _ => panic!("Invalid action"),
        }
    }
}

impl RandomWalkEnvironment {
    pub fn new(
        num_of_nodes: usize,
        start_node: usize,
        left_reward: f64,
        right_reward: f64,
    ) -> Self {
        Self {
            num_of_nodes,
            start_node,
            left_reward,
            right_reward,
        }
    }

    pub fn get_start_node(&self) -> RandomWalkNode {
        RandomWalkNode::new(self, self.start_node)
    }
}

pub struct RandomWalkAgent {
    state_values: HashMap<String, f64>,
    n: usize,
    discount_rate: f64,
    size_step_parameter: f64,
    policy: RandomPolicy,
    number_of_episodes: usize,
    environment: RandomWalkEnvironment,
}

impl RandomWalkAgent {
    pub fn new(
        n: usize,
        discount_rate: f64,
        size_step_parameter: f64,
        number_of_episodes: usize,
        environment: RandomWalkEnvironment,
    ) -> Self {
        let mut policy = RandomPolicy::new();

        (0..=environment.num_of_nodes).for_each(|i| {
            let state_id = i.to_string();
            let actions = vec![String::from("left"), String::from("right")];
            policy.set_actions_for_state(state_id, actions);
        });

        Self {
            state_values: HashMap::new(),
            n,
            discount_rate,
            size_step_parameter,
            policy,
            number_of_episodes,
            environment,
        }
    }

    pub fn run(&mut self) {
        (0..self.number_of_episodes).for_each(|_| {
            let mut state = self.environment.get_start_node();
            let mut next_state: Option<RandomWalkNode> = None;
            let mut terminal_time_step: Option<usize> = None;
            let mut current_time_step: usize = 0;
            let mut rewards: Vec<f64> = Vec::new();
            let mut previous_state_ids = vec![];

            loop {
                let current_state_id = state.get_id();
                previous_state_ids.push(current_state_id.clone());
                if current_time_step < terminal_time_step.unwrap_or(usize::MAX) {
                    let action = self.policy.select_action_for_state(&current_state_id).unwrap();
                    let (reward, ns) = state.take_action(&action);
                    rewards.push(reward);
                    if ns.is_terminal() {
                        terminal_time_step = Some(current_time_step + 1);
                    }
                    next_state = Some(ns);
                }

                let index_of_state_to_update = current_time_step as i32 - (self.n + 1) as i32;
                if index_of_state_to_update >= 0 {
                    let state_id_at_index_to_update = previous_state_ids[index_of_state_to_update as usize].clone();
                    let start_index = index_of_state_to_update as usize + 1;
                    let end_index: usize = index_of_state_to_update as usize + self.n + 1;
                    let end_index = end_index.min(terminal_time_step.unwrap_or(usize::MAX));
                    let mut discounted_sum_of_rewards = (start_index..=end_index).map(|i| {
                        let pow = i as i32 - index_of_state_to_update - 1;
                        self.discount_rate.powi(pow) * rewards[i]
                    }).sum::<f64>();
                    if index_of_state_to_update as usize + self.n < terminal_time_step.unwrap_or(usize::MAX) {
                        let state_id_at_index_plus_n = (index_of_state_to_update as usize + self.n).to_string();
                        let existing_value = self.state_values.get(&state_id_at_index_plus_n).unwrap_or(&0.0);
                        discounted_sum_of_rewards = discounted_sum_of_rewards + (self.discount_rate.powi(self.n as i32) * existing_value);
                    }
                    let existing_value = self.state_values.get(&index_of_state_to_update.to_string()).unwrap_or(&0.0);
                    let new_value = existing_value + (self.size_step_parameter * (discounted_sum_of_rewards - existing_value));
                    self.state_values.insert(state_id_at_index_to_update, new_value);
                }

                current_time_step += 1;
                state = next_state.clone().unwrap();
                if state.is_terminal() {
                    break;
                }
            }

        });
    }

    pub fn get_state_values(&self) -> &HashMap<String, f64> {
        &self.state_values
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multiple_n_steps_on_random_walk_environment() {
        let random_walk_environment = RandomWalkEnvironment::new(19, 9, -1.0, 0.0);
        let mut random_walk_agent = RandomWalkAgent::new(100, 1.0, 0.9, 100, random_walk_environment);
        random_walk_agent.run();
        let state_values = random_walk_agent.get_state_values();
        (0..19).for_each(|i| {
            let state_id = i.to_string();
            println!("State {}: {}", state_id, state_values.get(&state_id).unwrap_or(&0.0));
        })
    }
}
