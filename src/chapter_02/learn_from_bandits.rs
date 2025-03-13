use crate::chapter_02::k_armed_bandit::KArmedBandit;
use rand::Rng;

fn learn(bandits: &KArmedBandit, explore_rate: f32, steps: u32) -> Vec<(f32, f32)> {
    let num_of_bandits = bandits.get_bandit().len();
    let mut estimated_action_values: Vec<f32> = vec![0.0; num_of_bandits];

    let mut count_of_times_actions_taken: Vec<u32> = vec![0; num_of_bandits];
    let mut total_rewards_of_actions_taken: Vec<f32> = vec![0.0; num_of_bandits];

    let mut total_reward: f32 = 0.0;

    let mut data_for_graph = vec![(0f32, 0f32)];

    (0..steps).for_each(|i| {
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
            total_rewards_of_actions_taken[action_to_take_id],
        );

        total_reward += reward;

        let avarage_reward = total_reward / (i + 1) as f32;
        data_for_graph.push((i as f32, avarage_reward));
    });

    for (id, value) in estimated_action_values.iter().enumerate() {
        let actual = bandits.get_bandit()[id].get_reward();
        let diff = actual - value;
        println!(
            "action id: {},\testimated value: {},\tactual value: {},\tdiff: {}",
            id, value, actual, diff
        );
    }

    data_for_graph
}

fn calc_new_expected_action_value_old(
    count_of_times_actions_taken: u32,
    total_rewards_of_actions_taken: f32,
) -> f32 {
    total_rewards_of_actions_taken / count_of_times_actions_taken as f32
}

fn calc_new_expected_action_value_new(
    current_estimate: f32,
    count_of_times_actions_taken: u32,
    reward: f32,
) -> f32 {
    current_estimate + (1.0 / count_of_times_actions_taken as f32) * (reward - current_estimate)
}

fn calc_weighted_average(weight: f32, current_average: f32, new_reward: f32) -> f32 {
    current_average + weight * (new_reward - current_average)
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
    use crate::service::{LineChartBuilder, LineChartData};
    use plotters::prelude::{ShapeStyle, BLACK, BLUE, GREEN};
    use std::path::PathBuf;
    use std::time::Instant;

    fn setup_bandit() -> KArmedBandit {
        KArmedBandit::rand_new(10)
    }

    #[test]
    fn test_learn() {
        let bandit = setup_bandit();
        let a_return = learn(&bandit, 0.1, 1000);
        let a = LineChartData::new("e = 0.1".to_string(), a_return, ShapeStyle::from(&BLUE));

        let b_return = learn(&bandit, 0.01, 1000);
        let b = LineChartData::new("e = 0.01".to_string(), b_return, ShapeStyle::from(&GREEN));

        let c_return = learn(&bandit, 0.0, 1000);
        let c = LineChartData::new("e = 0".to_string(), c_return, ShapeStyle::from(&BLACK));

        let mut builder = LineChartBuilder::new();
        builder
            .add_data(a)
            .add_data(b)
            .add_data(c)
            .set_path(PathBuf::from(
                "output/chapter2/average_reward_by_explore_rate.png",
            ));
        builder.create_chart().unwrap();
    }

    #[test]
    fn test_averaging_functions() {
        let mut rng = rand::thread_rng();
        let random_floats: Vec<f32> = (0..10000000).map(|_| rng.random()).collect();
        let sum: f32 = random_floats.iter().sum();

        let existing_average = calc_new_expected_action_value_old(
            random_floats.len() as u32,
            random_floats.iter().sum::<f32>(),
        );

        let new_reward: f32 = rng.random();

        let start = Instant::now();

        println!(
            "existing average: {}, new reward: {}",
            existing_average, new_reward
        );

        let old_estimate =
            calc_new_expected_action_value_old(random_floats.len() as u32 + 1, sum + new_reward);

        let duration = start.elapsed(); // Measure the elapsed time
        println!("Time elapsed old method: {:?}", duration);

        let new_estimate = calc_new_expected_action_value_new(
            existing_average,
            random_floats.len() as u32 + 1,
            new_reward,
        );

        let duration = start.elapsed(); // Measure the elapsed time
        println!("Time elapsed new method: {:?}", duration);

        println!(
            "old estimate: {}, new estimate: {}",
            old_estimate, new_estimate
        );
        assert_eq!(new_estimate, old_estimate);
    }

    #[test]
    fn test_weighted_average() {
        let rewards = vec![100.1, 1.2, 1.2, 1.1, 1.3, 1.2, 0.9];
        let mut current_average: f32 = 0.0;
        let weight: f32 = 0.5;

        for reward in rewards.iter() {
            current_average = calc_weighted_average(weight, current_average, *reward);
            println!("current average: {}", current_average);
        }

        println!("final current average: {}", current_average);

        let mut accumulated_weighted_score: f32 = 0.0;
        for (n, reward) in rewards.iter().enumerate() {
            let a = (1.0 - weight).powi((rewards.len() - 1 - n) as i32);
            let b = weight * a;
            let c = b * reward;
            accumulated_weighted_score += c;
            println!("accumulated_weighted_score: {}", accumulated_weighted_score);
        }

        println!(
            "final accumulated_weighted_score: {}",
            accumulated_weighted_score
        );
    }
}
