use crate::attempts_at_framework::v1::policy::RandomPolicy;
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
    state_values: HashMap<usize, f64>,
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
        Self {
            state_values: HashMap::new(),
            n,
            discount_rate,
            size_step_parameter,
            policy: RandomPolicy::new(),
            number_of_episodes,
            environment,
        }
    }

    pub fn run(&mut self) {}

    pub fn get_state_values(&self) -> &HashMap<usize, f64> {
        &self.state_values
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multiple_n_steps_on_random_walk_environment() {
        let random_walk_environment = RandomWalkEnvironment::new(19, 9, -1.0, 0.0);
        let mut random_walk_agent = RandomWalkAgent::new(1, 1.0, 0.1, 10, random_walk_environment);
        random_walk_agent.run();
        println!("{:?}", random_walk_agent.get_state_values());
    }
}
