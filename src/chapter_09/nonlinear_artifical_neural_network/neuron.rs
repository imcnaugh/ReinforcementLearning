pub trait Neuron {
    fn forward(&self, inputs: &[f64]) -> f64;
}
