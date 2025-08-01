fn soft_max(preferences: &[f64]) -> Vec<f64> {
    let denominator = preferences.iter().fold(0.0, |acc, x| acc + x.exp());
    preferences.iter().map(|x| x.exp() / denominator).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax() {
        let preferences = vec![-3.0, -1.0];
        let softmax = soft_max(&preferences);
        let sum_softmax = softmax.iter().sum::<f64>();

        println!(
            "{:?}",
            preferences.iter().zip(softmax.iter()).collect::<Vec<_>>()
        );
        println!("{:?}", sum_softmax);

        assert_eq!(sum_softmax, 1.0);
    }

    #[test]
    fn test_softmax_more() {
        let preferences = vec![0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let softmax = soft_max(&preferences);
        let sum_softmax = softmax.iter().sum::<f64>();

        println!(
            "{:?}",
            preferences.iter().zip(softmax.iter()).collect::<Vec<_>>()
        );
        println!("{:?}", sum_softmax);

        assert!(sum_softmax - 1.0 < 0.00001);
    }
}
