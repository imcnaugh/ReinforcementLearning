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
    fn get_weights_and_bias(&self) -> (&[f64], &f64) {
        (self.weights.as_slice(), &self.bias)
    }

    fn forward(&self, inputs: &[f64]) -> f64 {
        assert_eq!(inputs.len(), self.weights.len());

        inputs
            .iter()
            .zip(self.weights.iter())
            .fold(self.bias, |acc, (x, w)| acc + x * w)
            .max(0.0)
    }

    fn backwards(&mut self, inputs: &[f64], expected: f64, learning_rate: f64) -> Vec<f64> {
        let current_value = self.forward(inputs);
        let derivative = if current_value > 0.0 { 1.0 } else { 0.0 };

        let gradient = learning_rate * derivative * expected;

        self.weights
            .iter_mut()
            .zip(inputs)
            .for_each(|(weight, input)| {
                *weight += gradient * input;
            });

        self.bias += gradient;

        self.weights
            .iter()
            .map(|weight| weight * derivative * expected)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_relu_neuron() {
        let mut neuron = ReluNeuron::new(2);
        let inputs = vec![1.0, 2.0];
        let expected = 1.0;
        let learning_rate = 0.01;

        let mut iteration_count: usize = 0;
        let mut output = Vec::new();
        for _ in 0..10000 {
            let calculated_output = neuron.forward(&inputs);
            if (calculated_output - expected).abs() < 0.000001 {
                break;
            }

            output = neuron.backwards(&inputs, expected, learning_rate);

            iteration_count += 1;
        }

        println!(
            "converged after {} iterations, with weights: {:?}",
            iteration_count,
            neuron.get_weights_and_bias()
        );
        println!("output: {:?}", output);
    }
}