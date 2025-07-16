fn lambda_return(lambda: f64, rewards: Vec<f64>) -> f64 {
    let mut sum = 0.0;

    for n in 1..rewards.len() {
        let n_step_return: f64 = rewards[0..n].iter().sum();
        sum += (1.0 - lambda) * lambda.powi((n - 1) as i32) * n_step_return;
    }

    let full_return: f64 = rewards.iter().sum();
    sum += lambda.powi((rewards.len() - 1) as i32) * full_return;

    sum
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
