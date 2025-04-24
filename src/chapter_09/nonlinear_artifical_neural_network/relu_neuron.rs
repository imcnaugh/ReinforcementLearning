use crate::chapter_09::nonlinear_artifical_neural_network::neuron::Neuron;

struct ReluNeuron {
    weights: Vec<f64>,
    bias: f64,
}

impl ReluNeuron {
    pub fn new(input_size: usize) -> Self {
        Self {
            weights: vec![0.0; input_size],
            bias: 0.0,
        }
    }
}

impl Neuron for ReluNeuron {
    fn forward(&self, inputs: &[f64]) -> f64 {
        assert_eq!(inputs.len(), self.weights.len());

        inputs
            .iter()
            .zip(self.weights.iter())
            .fold(self.bias, |acc, (x, w)| acc + x * w)
            .max(0.0)
    }
}
