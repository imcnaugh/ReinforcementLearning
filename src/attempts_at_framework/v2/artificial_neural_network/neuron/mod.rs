pub mod linear_neuron;
pub mod relu_neuron;

pub trait Neuron {
    fn new(number_of_inputs: usize) -> Self
    where
        Self: Sized;
    fn get_weights_and_bias(&self) -> (&[f64], &f64);
    fn forward(&self, inputs: &[f64]) -> f64;
    fn backwards(&mut self, inputs: &[f64], gradient: f64, learning_rate: f64) -> Vec<f64>;
    fn activation_derivative(&self, output: f64) -> f64;
}
