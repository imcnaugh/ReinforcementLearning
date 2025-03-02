#[cfg(test)]
mod tests {
    #[test]
    fn exercise_3_10() {
        const DISCOUNT_RATE: f32 = 0.5f32;

        let a: f32 = (0..100).map(|i| DISCOUNT_RATE.powi(i)).sum();
        let b = 1f32 / (1f32 - DISCOUNT_RATE);

        println!("a: {}, b: {}", a, b);
        assert_eq!(a, b);
    }

    fn discount_reward_primitive(rewards: &Vec<f32>, discount_rate: f32) -> f32 {
        rewards.iter().enumerate().fold(0.0, |acc, (i, r)| {
            let reward_with_discount = discount_rate.powi(i as i32) * r;
            acc + reward_with_discount
        })
    }

    /// Computes the total discounted reward using dynamic programming.
    ///
    /// **Note:** This implementation follows a simplistic approach, as suggested
    /// by ChatGPT, to illustrate discounting rewards. It's functional for learning
    /// but may not capture best practices for optimal performance or code design.
    ///
    /// You should revisit this function after completing Chapter 4 of the book,
    /// where the "real way" to compute discounted rewards with more efficient
    /// techniques is explained.
    fn discount_reward_dynamic_programming(rewards: &Vec<f32>, discount_rate: f32) -> f32 {
        let g = 0f32;
        let mut returns = vec![];

        for r in rewards.iter().rev() {
            let g = r + discount_rate * g;
            returns.push(g);
        }

        returns[returns.len() - 1]
    }

    #[test]
    fn discount_reward_test() {
        let discount_rate = 0.9f32;
        let rewards_all_zeros = vec![0.0, 0.0, 0.0, 0.0, 0.0];
        let all_zeros_total_discounted =
            discount_reward_primitive(&rewards_all_zeros, discount_rate);
        assert_eq!(all_zeros_total_discounted, 0.0);

        let random_rewards = (0..100).map(|_| rand::random::<f32>()).collect();
        let random_rewards_total_discounted =
            discount_reward_primitive(&random_rewards, discount_rate);
        let sum_of_random_rewards = &random_rewards.iter().sum::<f32>();
        println!(
            "random_rewards_total_discounted: {}",
            random_rewards_total_discounted
        );
        println!("sum_of_random_rewards: {}", sum_of_random_rewards);
        assert!(random_rewards_total_discounted > 0.0);

        let possible = discount_reward_dynamic_programming(&random_rewards, discount_rate);

        println!("possible: {}", possible);
    }
}
