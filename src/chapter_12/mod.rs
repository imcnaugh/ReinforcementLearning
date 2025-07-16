fn lambda_return(lambda: f64, rewards: Vec<f64>) -> f64 {
    let split_last = rewards.split_last();
    match split_last {
        Some((last, rest)) => {
            let sum = rest
                .iter()
                .enumerate()
                .map(|(i, r)| single_lambda_return(lambda, i + 1, *r))
                .sum::<f64>();

            (1.0 - lambda) * sum + lambda * single_lambda_return(lambda, rest.len(), *last)
        }
        None => 0.0,
    }
}

fn single_lambda_return(lambda: f64, n: usize, reward: f64) -> f64 {
    let exponent = (n - 1) as f64;
    lambda.powf(exponent) * reward
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_0_step_lambda_return() {
        let rewards = vec![200.0, 400.0, 6.32];
        let lambda = 0.0;
        let actual = lambda_return(lambda, rewards);
        let expected = 1.0;
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_1_step_lambda_return() {
        let rewards = vec![0.0, 200.0];
        let lambda = 1.0;
        let actual = lambda_return(lambda, rewards);
        let expected = 200.0;
        assert_eq!(actual, expected);
    }
}
