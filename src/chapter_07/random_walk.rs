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
        if self.environment.num_of_nodes - 1 == self.id {
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
                let reward = if self.id == 1 {
                    self.environment.left_reward
                } else {
                    0.0
                };
                (reward, RandomWalkNode::new(self.environment, self.id - 1))
            }
            "right" => {
                let reward = if self.id == self.environment.num_of_nodes - 2 {
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

        (0..environment.num_of_nodes).for_each(|i| {
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
                    let action = self
                        .policy
                        .select_action_for_state(&current_state_id)
                        .unwrap();
                    let (reward, ns) = state.take_action(&action);
                    rewards.push(reward);
                    if ns.is_terminal() {
                        terminal_time_step = Some(current_time_step + 1);
                    }
                    next_state = Some(ns);
                }

                let index_of_state_to_update = current_time_step as i32 - (self.n + 1) as i32;
                if index_of_state_to_update >= 0 {
                    let state_id_at_index_to_update =
                        previous_state_ids[index_of_state_to_update as usize].clone();
                    let start_index = index_of_state_to_update as usize + 1;
                    let end_index: usize = index_of_state_to_update as usize + self.n;
                    let end_index = end_index.min(terminal_time_step.unwrap_or(usize::MAX));
                    let mut discounted_sum_of_rewards = (start_index..=end_index)
                        .map(|i| {
                            let pow = i as i32 - index_of_state_to_update - 1;
                            self.discount_rate.powi(pow) * rewards[i - 1]
                        })
                        .sum::<f64>();
                    if index_of_state_to_update as usize + self.n
                        < terminal_time_step.unwrap_or(usize::MAX)
                    {
                        let state_id_at_index_plus_n =
                            (index_of_state_to_update as usize + self.n).to_string();
                        let existing_value = self
                            .state_values
                            .get(&state_id_at_index_plus_n)
                            .unwrap_or(&0.0);
                        discounted_sum_of_rewards = discounted_sum_of_rewards
                            + (self.discount_rate.powi(self.n as i32) * existing_value);
                    }
                    let existing_value = self
                        .state_values
                        .get(&state_id_at_index_to_update)
                        .unwrap_or(&0.0);
                    let new_value = existing_value
                        + (self.size_step_parameter * (discounted_sum_of_rewards - existing_value));
                    self.state_values
                        .insert(state_id_at_index_to_update, new_value);
                }

                current_time_step += 1;
                state = next_state.clone().unwrap();
                if index_of_state_to_update
                    == (terminal_time_step.unwrap_or(i32::MAX as usize) as i32 - 1)
                {
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
    use crate::attempts_at_framework::v1::agent::NStepSarsa;
    use crate::service::{LineChartBuilder, LineChartData};
    use plotters::prelude::{ShapeStyle, BLACK};

    #[test]
    fn test_multiple_n_steps_on_random_walk_environment() {
        let mut state_values: Vec<HashMap<String, f64>> = Vec::new();
        let size_step_parameters = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let repetitions = 100;
        let n = 4;
        let discount_rate = 1.0;
        let number_of_episodes = 10;
        let number_of_nodes = 19;
        let left_reward = -1.0;
        let right_reward = 1.0;
        let mut points: Vec<(f32, f32)> = Vec::new();
        size_step_parameters.iter().for_each(|size_step_parameter| {
            (0..repetitions).for_each(|_| {
                let random_walk_environment = RandomWalkEnvironment::new(
                    number_of_nodes,
                    number_of_nodes / 2,
                    left_reward,
                    right_reward,
                );
                let mut random_walk_agent = RandomWalkAgent::new(
                    n,
                    discount_rate,
                    size_step_parameter.clone(),
                    number_of_episodes,
                    random_walk_environment,
                );
                random_walk_agent.run();
                state_values.push(random_walk_agent.get_state_values().clone())
            });
            let mean_squared_error = (0..number_of_nodes)
                .map(|i| {
                    let state_values = state_values.iter().fold(0.0, |mut acc, x| {
                        acc + x.get(&i.to_string()).unwrap_or(&0.0)
                    });
                    let state_id = i.to_string();
                    let expected_value = if i == 0 || i == number_of_nodes - 1 {
                        0.0
                    } else {
                        (((right_reward - left_reward) / (number_of_nodes - 1) as f64) * i as f64)
                            + left_reward
                    };
                    let average_value = state_values / repetitions as f64;
                    let error = average_value - expected_value;
                    // println!("State {}: value: {} expected: {}, diff: {}", state_id, average_value, expected_value, error);
                    error.powi(2)
                })
                .sum::<f64>();
            let average_mean_squared_error = mean_squared_error / number_of_nodes as f64;
            points.push((
                size_step_parameter.clone() as f32,
                average_mean_squared_error as f32,
            ));
            println!("Mean squared error: {}", average_mean_squared_error.sqrt());
        });

        let line_chart_points =
            LineChartData::new(format!("{} step mse", n), points, ShapeStyle::from(&BLACK));
        let mut builder = LineChartBuilder::new();
        builder
            .set_path(std::path::PathBuf::from(
                "output/chapter7/random_walk_mse.png",
            ))
            .add_data(line_chart_points);
        builder.create_chart().unwrap();
    }

    #[test]
    fn test_n_step_agent_v1() {
        let number_of_nodes = 19;
        let left_reward = -1.0;
        let right_reward = 1.0;

        let random_walk_environment = RandomWalkEnvironment::new(
            number_of_nodes,
            number_of_nodes / 2,
            left_reward,
            right_reward,
        );

        let starting_state = random_walk_environment.get_start_node();

        let mut agent = NStepSarsa::new(4, 0.0, 0.4, 1.0);
        agent.learn_for_episode_count(1, vec![starting_state.clone().clone()]);

        let policy = agent.get_policy();

        println!("Policy");
    }
}
