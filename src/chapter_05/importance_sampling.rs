use crate::chapter_05::policy::Policy;
use crate::service::calc_average;

/// # Ordinary importance sampling
///
/// from page 104 of the book
///
/// ## Args
/// runs: a vector of tuples, the first element of the tuple is the state actions pairs taken, and the second element is the reward. //TODO find a better way to represent all of this
/// target_policy
/// behavior_policy
pub fn ordinary_importance_sampling<TP: Policy, BP: Policy>(
    runs: &Vec<(&Vec<(&str, &str)>, f64)>,
    target_policy: &TP,
    behavior_policy: &BP,
) -> Result<f64, String> {
    runs.iter()
        .enumerate()
        .try_fold(0.0, |current_average, (index, run)| {
            let iteration = index + 1;
            let state_action_pairs = run.0;
            let reward = run.1;
            let importance_ratio = calculate_importance_sampling_ratio(
                state_action_pairs,
                target_policy,
                behavior_policy,
            )?;
            let adjusted_reward = reward * importance_ratio;

            Ok(calc_average(
                current_average,
                iteration as i32,
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
    runs: &Vec<(&Vec<(&str, &str)>, f64)>,
    target_policy: &TP,
    behavior_policy: &BP,
) -> Result<f64, String> {
    let mut numerator = 0.0;
    let mut denominator = 0.0;

    runs.iter().try_for_each(|run| {
        let importance_sampling_ratio =
            calculate_importance_sampling_ratio(run.0, target_policy, behavior_policy)?;
        numerator += run.1 * importance_sampling_ratio;
        denominator += importance_sampling_ratio;
        Ok::<(), String>(())
    })?;

    if denominator == 0.0 {
        return Ok(0.0);
    }

    Ok(numerator / denominator)
}

pub fn calculate_importance_sampling_ratio<TP: Policy, BP: Policy>(
    state_action_paris: &Vec<(&str, &str)>,
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
    state_action_pair: &(&str, &str),
) -> Result<f64, String> {
    let state_id = state_action_pair.0;
    let action_id = state_action_pair.1;
    match policy.get_actions_for_state(state_id) {
        Ok(actions) => match actions.iter().find(|&a| a.1.eq(action_id)) {
            None => Err("action not found".to_string()),
            Some((odds, _)) => Ok(*odds),
        },
        Err(_) => Err("state not found".to_string()),
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

        let runs = vec![(&state_action_pairs, 10.0)];

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

        let runs = vec![(&state_action_pairs, 10.0)];

        let weighted_importance_sampling =
            weighted_importance_sampling(&runs, &target_policy, &behavior_policy);

        assert_eq!(weighted_importance_sampling.unwrap(), 10.0);
    }
}
