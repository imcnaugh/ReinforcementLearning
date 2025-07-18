#[cfg(test)]
mod tests {

    #[test]
    fn test_lambda_rate_summing_to_1() {
        let lambda_rate: f64 = 0.02;

        let max = 20;
        let powers = (1..max)
            .map(|i| lambda_rate.powi(i - 1))
            .collect::<Vec<f64>>();
        let sum = powers.iter().sum::<f64>() * (1.0 - lambda_rate);
        let remainder = max - 1;
        let remainder_power = lambda_rate.powi(remainder);

        println!("{:?}", powers);
        println!("{:?}", sum);
        println!("{:?}", remainder_power);
        let total = sum + remainder_power;
        println!("{:?}", total);
        assert_eq!(total, 1.0);
    }
}
