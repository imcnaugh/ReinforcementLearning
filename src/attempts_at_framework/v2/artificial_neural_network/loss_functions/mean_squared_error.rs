use crate::attempts_at_framework::v2::artificial_neural_network::loss_functions::LossFunction;

pub struct MeanSquaredError;

impl LossFunction for MeanSquaredError {
    fn calculate_loss(&self, expected: &[f64], predicted: &[f64]) -> f64 {
        expected
            .iter()
            .zip(predicted.iter())
            .map(|(e, p)| (e - p).powi(2))
            .sum()
    }

    fn calculate_gradient(&self, expected: &[f64], predicted: &[f64]) -> Vec<f64> {
        expected
            .iter()
            .zip(predicted.iter())
            .map(|(e, p)| e - p)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_loss_zero_loss() {
        let expected = vec![1.0, 2.0, 3.0];
        let predicted = vec![1.0, 2.0, 3.0];
        let loss = MeanSquaredError.calculate_loss(&expected, &predicted);
        assert_eq!(loss, 0.0);
    }

    #[test]
    fn test_calculate_loss_non_zero_loss() {
        let expected = vec![1.0, 2.0, 3.0];
        let predicted = vec![-1.0, 1.0, 5.0];
        let loss = MeanSquaredError.calculate_loss(&expected, &predicted);
        assert_eq!(loss, 9.0);
    }
}
