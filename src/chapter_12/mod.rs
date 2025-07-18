mod general_learnings;
mod true_td_lambda;

fn lambda_return(lambda: f64, rewards: Vec<f64>) -> f64 {
    let mut running_reward_total = rewards[0];
    let sum = rewards[1..].iter().enumerate().fold(0.0, |acc, (i, r)| {
        let weighted_reward_total = (1.0 - lambda) * lambda.powi(i as i32) * running_reward_total;
        running_reward_total += r;
        acc + weighted_reward_total
    });

    let weighted_reward_sum = lambda.powi((rewards.len() - 1) as i32) * running_reward_total;
    sum + weighted_reward_sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_0_step_lambda_return() {
        let rewards = vec![200.0, 400.0, 6.32];
        let lambda = 0.0;
        let actual = lambda_return(lambda, rewards);
        let expected = 200.0;
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_full_lambda_return() {
        let rewards = vec![1.0, 200.0];
        let lambda = 1.0;
        let actual = lambda_return(lambda, rewards);
        let expected = 201.0;
        assert_eq!(actual, expected);
    }
}
