fn lambda_return(lambda: f64, rewards: Vec<f64>) -> f64 {
    if rewards.is_empty() {
        return 0.0;
    }

    let t = 0; // Start time
    let T = rewards.len(); // Terminal time

    // Handle special cases first
    if lambda == 1.0 {
        // For λ = 1, return the full return (Monte Carlo case)
        return rewards.iter().sum();
    }
    if lambda == 0.0 {
        // For λ = 0, return one-step return
        return rewards[0];
    }

    // Implement equation 12.3:
    // Gₜλ = (1-λ)Σₙ₌₁ᵀ⁻ᵗ⁻¹ λⁿ⁻¹Gₜ:ₜ₊ₙ + λᵀ⁻ᵗ⁻¹Gₜ

    let mut sum = 0.0;

    // Calculate the main sum (first term)
    for n in 1..T {
        let n_step_return: f64 = rewards[0..n].iter().sum();
        sum += (1.0 - lambda) * lambda.powi((n - 1) as i32) * n_step_return;
    }

    // Add the final term (terminal backup)
    let full_return: f64 = rewards.iter().sum();
    sum += lambda.powi((T - 1) as i32) * full_return;

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
