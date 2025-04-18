use crate::chapter_05::policy::Policy;
use crate::service::calc_average;
use std::fmt::format;

/// # Ordinary importance sampling
///
/// from page 104 of the book
///
/// ## Args
/// runs: a vector of tuples, the first element of the tuple is the state actions pairs taken, and the second element is the reward. //TODO find a better way to represent all of this
/// target_policy
/// behavior_policy
pub fn ordinary_importance_sampling<TP: Policy, BP: Policy>(
    runs: &Vec<(Vec<(String, String)>, f64)>,
    target_policy: &TP,
    behavior_policy: &BP,
) -> Result<f64, String> {
    runs.iter()
        .enumerate()
        .try_fold(0.0, |current_average, (index, run)| {
            let importance_ratio =
                calculate_importance_sampling_ratio(&run.0, target_policy, behavior_policy)?;
            let adjusted_reward = run.1 * importance_ratio;

            Ok(calc_average(
                current_average,
                (index + 1) as i32,
                adjusted_reward,
            ))
        })
}

/// # Weighted importance sampling
///
/// from page 105 of the book
///
/// ## Args
/// runs: a vector of tuples, the first element of the tuple is the state actions pairs taken, and the second element is the reward. //TODO find a better way to represent all of this
/// target_policy
/// behavior_policy
pub fn weighted_importance_sampling<TP: Policy, BP: Policy>(
    runs: &Vec<(Vec<(String, String)>, f64)>,
    target_policy: &TP,
    behavior_policy: &BP,
) -> Result<f64, String> {
    let mut numerator = 0.0;
    let mut denominator = 0.0;

    runs.iter().try_for_each(|run| {
        match calculate_importance_sampling_ratio(&run.0, target_policy, behavior_policy) {
            Ok(importance_sampling_ratio) => {
                numerator += run.1 * importance_sampling_ratio;
                denominator += importance_sampling_ratio;
                Ok::<(), String>(())
            }
            Err(_) => Ok::<(), String>(()),
        }
    })?;

    if denominator == 0.0 {
        return Ok(0.0);
    }

    Ok(numerator / denominator)
}

pub fn weighted_importance_sampling_incremental<TP: Policy, BP: Policy>(
    new_state_action_pairs: &Vec<(String, String)>, // State-action pairs for the new episode
    new_reward: f64,                                // Return (G_n) for the new episode
    current_average: f64,                           // Current value estimate (V_n)
    cumulative_sum_of_weights: Option<f64>,         // Cumulative sum of weights (C_n)
    target_policy: &TP,                             // Target policy
    behavior_policy: &BP,                           // Behavior policy
) -> Result<(f64, Option<f64>), String> {
    // Returns (new_average, new_cumulative_sum)
    // Calculate the importance sampling ratio (W_n) for the new episode
    let importance_sampling_ratio = calculate_importance_sampling_ratio(
        new_state_action_pairs,
        target_policy,
        behavior_policy,
    )?;

    // Update the cumulative sum of weights (C_{n+1} = C_n + W_n)
    let new_cumulative_sum = cumulative_sum_of_weights.unwrap_or(0.0) + importance_sampling_ratio;

    // Update the value estimate (V_{n+1})
    let new_average = match cumulative_sum_of_weights {
        None => new_reward,
        Some(_) => {
            current_average
                + (importance_sampling_ratio / new_cumulative_sum) * (new_reward - current_average)
        }
    };

    // Return the new average and cumulative sum
    Ok((new_average, Some(new_cumulative_sum)))
}

pub fn calculate_importance_sampling_ratio<TP: Policy, BP: Policy>(
    state_action_paris: &Vec<(String, String)>,
    target_policy: &TP,
    behavior_policy: &BP,
) -> Result<f64, String> {
    state_action_paris
        .iter()
        .try_fold(1.0, |acc, state_action_pair| {
            let numerator =
                find_odds_of_taking_action_at_state_for_policy(target_policy, state_action_pair)?;
            let denominator =
                find_odds_of_taking_action_at_state_for_policy(behavior_policy, state_action_pair)?;
            Ok(acc * (numerator / denominator))
        })
}

fn find_odds_of_taking_action_at_state_for_policy<TP: Policy>(
    policy: &TP,
    state_action_pair: &(String, String),
) -> Result<f64, String> {
    let state_id = &state_action_pair.0;
    let action_id = &state_action_pair.1;
    match policy.get_actions_for_state(state_id) {
        Ok(actions) => match actions.iter().find(|&a| a.1.eq(action_id)) {
            Some((odds, _)) => Ok(*odds),
            None => Ok(0.0),
        },
        Err(_) => Err(format!("could not find state: {}", state_id)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chapter_05::policy::{DeterministicPolicy, StochasticPolicy};

    #[test]
    fn test_calculate_importance_sampling_ratio() {
        let state_action_pairs = vec![("state_1", "state_1 -> state_2")];

        let mut target_policy = DeterministicPolicy::new();
        target_policy.set_action_for_state(&state_action_pairs[0].0, state_action_pairs[0].1);

        let mut behavior_policy = StochasticPolicy::new();
        behavior_policy
            .set_state_actions_probabilities_using_e_soft_probabilities(
                state_action_pairs[0].0,
                vec![
                    String::from(state_action_pairs[0].1),
                    String::from("some_other_action"),
                ],
                1.0, // 1 is impractical, but it makes the math clean for testing.
                state_action_pairs[0].1.to_string(),
            )
            .unwrap();
        let state_action_pairs: Vec<(String, String)> = state_action_pairs
            .iter()
            .map(|x| (x.0.to_string(), x.1.to_string()))
            .collect();

        let importance_ratio = calculate_importance_sampling_ratio(
            &state_action_pairs,
            &target_policy,
            &behavior_policy,
        );

        assert_eq!(importance_ratio.unwrap(), 2.0);
    }

    #[test]
    fn test_ordinary_importance_sampling() {
        let state_action_pairs = vec![("state_1", "state_1 -> state_2")];

        let mut target_policy = DeterministicPolicy::new();
        target_policy.set_action_for_state(&state_action_pairs[0].0, state_action_pairs[0].1);

        let mut behavior_policy = StochasticPolicy::new();
        behavior_policy
            .set_state_actions_probabilities_using_e_soft_probabilities(
                state_action_pairs[0].0,
                vec![
                    String::from(state_action_pairs[0].1),
                    String::from("some_other_action"),
                ],
                1.0, // 1 is impractical, but it makes the math clean for testing.
                state_action_pairs[0].1.to_string(),
            )
            .unwrap();

        let state_action_pairs: Vec<(String, String)> = state_action_pairs
            .iter()
            .map(|x| (x.0.to_string(), x.1.to_string()))
            .collect();
        let runs = vec![(state_action_pairs, 10.0)];

        let ordinary_importance_sampling =
            ordinary_importance_sampling(&runs, &target_policy, &behavior_policy);

        assert_eq!(ordinary_importance_sampling.unwrap(), 20.0);
    }

    #[test]
    fn test_weighted_importance_sampling() {
        let state_action_pairs = vec![("state_1", "state_1 -> state_2")];

        let mut target_policy = DeterministicPolicy::new();
        target_policy.set_action_for_state(&state_action_pairs[0].0, state_action_pairs[0].1);

        let mut behavior_policy = StochasticPolicy::new();
        behavior_policy
            .set_state_actions_probabilities_using_e_soft_probabilities(
                state_action_pairs[0].0,
                vec![
                    String::from(state_action_pairs[0].1),
                    String::from("some_other_action"),
                ],
                1.0, // 1 is impractical, but it makes the math clean for testing.
                state_action_pairs[0].1.to_string(),
            )
            .unwrap();

        let state_action_pairs: Vec<(String, String)> = state_action_pairs
            .iter()
            .map(|x| (x.0.to_string(), x.1.to_string()))
            .collect();
        let runs = vec![(state_action_pairs, 10.0)];

        let weighted_importance_sampling =
            weighted_importance_sampling(&runs, &target_policy, &behavior_policy);

        assert_eq!(weighted_importance_sampling.unwrap(), 10.0);
    }

    #[test]
    fn compare_weighted_importance_sampling_methods() {
        let state_action_pairs_and_reward_1 = (
            vec![
                ("s1".to_string(), "s1>s2".to_string()),
                ("s2".to_string(), "s2>s3".to_string()),
                ("s3".to_string(), "s3>t".to_string()),
            ],
            10.0,
        );

        let state_action_pairs_and_reward_2 = (
            vec![
                ("s1".to_string(), "s1>s2".to_string()),
                ("s2".to_string(), "s2>s3".to_string()),
                ("s3".to_string(), "s3>t".to_string()),
            ],
            0.0,
        );

        let state_action_pairs_and_reward_3 = (
            vec![
                ("s1".to_string(), "s1>s2".to_string()),
                ("s2".to_string(), "s2>s3".to_string()),
                ("s3".to_string(), "s3>t".to_string()),
            ],
            10.0,
        );

        let state_action_pairs_and_reward_4 = (
            vec![
                ("s1".to_string(), "s1>s2".to_string()),
                ("s2".to_string(), "s2>t".to_string()),
            ],
            0.0,
        );

        let mut target_policy = DeterministicPolicy::new();
        target_policy.set_action_for_state("s1", "s1>s2");
        target_policy.set_action_for_state("s2", "s2>s3");
        target_policy.set_action_for_state("s3", "s3>t");

        let mut behavior_policy = StochasticPolicy::new();
        behavior_policy
            .set_state_action_probabilities("s1", vec![(1.0, "s1>s2".to_string())])
            .unwrap();
        behavior_policy
            .set_state_action_probabilities(
                "s2",
                vec![(0.9, "s2>s3".to_string()), (0.1, "s2>t".to_string())],
            )
            .unwrap();
        behavior_policy
            .set_state_action_probabilities(
                "s3",
                vec![(0.3, "s3>t".to_string()), (0.7, "s3>t".to_string())],
            )
            .unwrap();

        let episodes = vec![
            state_action_pairs_and_reward_2,
            state_action_pairs_and_reward_1,
            state_action_pairs_and_reward_4,
            // state_action_pairs_and_reward_3,
        ];
        let original_weighted_importance_value =
            weighted_importance_sampling(&episodes, &target_policy, &behavior_policy).unwrap();

        println!(
            "original weighted importance value: {}",
            original_weighted_importance_value
        );

        let mut current_average_with_new_method = 0.0;
        let mut cumulative_sum_of_weights: Option<f64> = None;

        for run in episodes {
            let (new_average, new_cumulative_sum) = weighted_importance_sampling_incremental(
                &run.0, // State-action pairs for the episode
                run.1,  // Reward for the episode
                current_average_with_new_method,
                cumulative_sum_of_weights,
                &target_policy,
                &behavior_policy,
            )
            .unwrap();

            // Update the cumulative sum of weights and the current average
            cumulative_sum_of_weights = new_cumulative_sum;
            current_average_with_new_method = new_average;
        }

        println!(
            "Final weighted importance sampling estimate: {}",
            current_average_with_new_method
        );

        println!(
            "new weighted importance value: {}",
            current_average_with_new_method
        );

        let diff = (original_weighted_importance_value - current_average_with_new_method).abs();
        assert!(diff < 0.00000000001);
    }
}
