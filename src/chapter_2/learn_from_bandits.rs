use rand::Rng;
use crate::chapter_2::k_armed_bandit::KArmedBandit;

fn learn(bandits: &KArmedBandit, explore_rate: f32, steps: u32) -> f32 {
    let num_of_bandits = bandits.get_bandit().len();
    let mut estimated_action_values: Vec<f32> = vec![0.0; num_of_bandits];

    let mut count_of_times_actions_taken: Vec<u32> = vec![0; num_of_bandits];
    let mut total_rewards_of_actions_taken: Vec<f32> = vec![0.0; num_of_bandits];
    
    let mut total_reward: f32 = 0.0;

    (0 .. steps).for_each(|i| {
        let action_to_take_id: usize = if rand::random::<f32>() < explore_rate {
            get_explore_action_id(num_of_bandits)
        } else {
            get_exploit_action_id(&estimated_action_values)
        };

        let reward = bandits.get_bandit()[action_to_take_id].get_reward();

        count_of_times_actions_taken[action_to_take_id] += 1;
        total_rewards_of_actions_taken[action_to_take_id] += reward;

        estimated_action_values[action_to_take_id] = calc_new_expected_action_value_old(
            count_of_times_actions_taken[action_to_take_id],
            total_rewards_of_actions_taken[action_to_take_id]);

        total_reward += reward;
    });

    for (id, value) in estimated_action_values.iter().enumerate() {
        let actual = bandits.get_bandit()[id].get_reward();
        let diff = actual - value;
        println!("action id: {},\testimated value: {},\tactual value: {},\tdiff: {}", id, value, actual, diff);
    }
    
    total_reward
}

fn calc_new_expected_action_value_old(count_of_times_actions_taken: u32, total_rewards_of_actions_taken: f32) -> f32 {
    total_rewards_of_actions_taken / count_of_times_actions_taken as f32
}

fn calc_new_expected_action_value_new(current_estimate: f32, count_of_times_actions_taken: u32, reward: f32) -> f32 {
    current_estimate + (1.0 / count_of_times_actions_taken as f32) * (reward - current_estimate)
}

fn get_explore_action_id(num_of_bandits: usize) -> usize {
    rand::rng().random_range(0..num_of_bandits)
}

fn get_exploit_action_id(estimated_action_values: &Vec<f32>) -> usize {
    let mut action_id: usize = 0;
    let mut current_max: f32 = f32::MIN;

    for (id, value) in estimated_action_values.iter().enumerate() {
        if *value > current_max {
            current_max = *value;
            action_id = id;
        }
    }
    
    action_id
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn setup_bandit() -> KArmedBandit {
        KArmedBandit::rand_new(10)
    }

    #[test]
    fn test_learn() {
        let bandit = setup_bandit();
        let total_reward = learn(&bandit, 0.1, 10000);
        println!("explore rate: {}, total reward: {}", 0.1, total_reward);


        let total_reward = learn(&bandit, 0.01, 10000);
        println!("explore rate: {}, total reward: {}", 0.01, total_reward);


        let total_reward = learn(&bandit, 0.0, 10000);
        println!("explore rate: {}, total reward: {}", 0.0, total_reward);
    }

    #[test]
    fn test_averaging_functions() {
        let mut rng = rand::thread_rng();
        let random_floats: Vec<f32> = (0..10000000).map(|_| rng.random()).collect();
        let sum: f32 = random_floats.iter().sum();

        let existing_average = calc_new_expected_action_value_old(random_floats.len() as u32, random_floats.iter().sum::<f32>());

        let new_reward: f32 = rng.random();

        let start = Instant::now();

        println!("existing average: {}, new reward: {}", existing_average, new_reward);

        let old_estimate = calc_new_expected_action_value_old(
            random_floats.len() as u32 + 1,
            sum + new_reward
        );

        let duration = start.elapsed(); // Measure the elapsed time
        println!("Time elapsed old method: {:?}", duration);

        let new_estimate = calc_new_expected_action_value_new(
            existing_average,
            random_floats.len() as u32 + 1,
            new_reward
        );

        let duration = start.elapsed(); // Measure the elapsed time
        println!("Time elapsed new method: {:?}", duration);

        println!("old estimate: {}, new estimate: {}", old_estimate, new_estimate);
        assert_eq!(new_estimate, old_estimate);
    }
}