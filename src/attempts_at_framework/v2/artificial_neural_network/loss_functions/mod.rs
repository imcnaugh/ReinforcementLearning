pub mod mean_squared_error;

pub trait LossFunction {
    fn calculate_loss(&self, expected: &[f64], predicted: &[f64]) -> f64;
    fn calculate_gradient(&self, expected: &[f64], predicted: &[f64]) -> Vec<f64>;
}
