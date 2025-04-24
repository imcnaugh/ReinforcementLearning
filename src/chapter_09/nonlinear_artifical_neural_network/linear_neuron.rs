use crate::chapter_09::nonlinear_artifical_neural_network::neuron::Neuron;

struct LinearNeuron {
    weights: Vec<f64>,
    bias: f64,
}

impl LinearNeuron {
    pub fn new(number_of_inputs: usize) -> Self {
        Self {
            weights: vec![0.0; number_of_inputs],
            bias: 0.0,
        }
    }
}

impl Neuron for LinearNeuron {
    fn forward(&self, inputs: &[f64]) -> f64 {
        assert_eq!(inputs.len(), self.weights.len());
        inputs
            .iter()
            .zip(self.weights.iter())
            .fold(self.bias, |acc, (x, w)| acc + x * w)
    }
}
