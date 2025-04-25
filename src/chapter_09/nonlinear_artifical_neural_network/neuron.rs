pub trait Neuron {
    fn get_weights_and_bias(&self) -> (&[f64], &f64);
    fn forward(&self, inputs: &[f64]) -> f64;
}
