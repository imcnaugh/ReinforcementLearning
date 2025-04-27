use crate::attempts_at_framework::v2::artificial_neural_network::neuron::Neuron;
use rand::{rng, Rng};

pub struct ReluNeuron {
    weights: Vec<f64>,
    bias: f64,
}

impl ReluNeuron {
    pub fn new(input_size: usize) -> Self {
        Self {
            weights: (0..input_size)
                .map(|_| rng().random_range(0.1..0.9))
                .collect(),
            bias: 0.0,
        }
    }

    pub fn build(weights: &[f64], bias: f64) -> Result<Self, Box<(dyn std::error::Error)>> {
        let neuron = Self {
            weights: weights.to_vec(),
            bias,
        };
        Ok(neuron)
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

    fn backwards(&mut self, inputs: &[f64], gradient: f64, learning_rate: f64) -> Vec<f64> {
        let current_value = self.forward(inputs);
        let derivative = if current_value > 0.0 { 1.0 } else { 0.0 };

        self.weights
            .iter_mut()
            .zip(inputs)
            .for_each(|(weight, input)| {
                *weight += derivative * gradient * input * learning_rate;
            });

        self.bias += gradient * derivative * learning_rate;

        self.weights
            .iter()
            .map(|weight| weight * derivative * gradient)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu_basic_functionality() {
        let neuron = ReluNeuron::build(&[0.5, -0.5], 0.0).unwrap();

        // Test positive output
        let inputs = vec![2.0, 1.0];
        // Expected calculation: (2.0 * 0.5) + (1.0 * -0.5) + 0.0 = 0.5
        let output = neuron.forward(&inputs);
        assert_eq!(output, 0.5);

        // Test negative input (should return 0)
        let inputs = vec![-2.0, 1.0];
        // Expected calculation: (-2.0 * 0.5) + (1.0 * -0.5) + 0.0 = -1.5
        // But ReLU should clamp it to 0
        let output = neuron.forward(&inputs);
        assert_eq!(output, 0.0);
    }

    #[test]
    fn test_relu_learning() {
        let mut neuron = ReluNeuron::build(&[0.1, 0.1], 0.0).unwrap();

        let inputs = vec![1.0, 1.0];
        let initial_output = neuron.forward(&inputs);

        // Train the neuron
        let gradients = neuron.backwards(&inputs, 1.0, 0.1);

        println!("{:?}", gradients);

        let final_output = neuron.forward(&inputs);
        assert!(
            final_output > initial_output,
            "Neuron should learn and increase output"
        );
    }

    #[test]
    fn test_convergence() {
        let mut neuron = ReluNeuron::build(&[0.1, 0.1], 0.0).unwrap();
        let inputs = vec![2.0, 1.0];
        let expected = 2.0;
        let learning_rate = 0.1;

        let mut iteration_count: usize = 0;
        let mut output = Vec::new();
        for _ in 0..10000 {
            let calculated_output = neuron.forward(&inputs);
            if (calculated_output - expected).abs() < 0.000001 {
                break;
            }

            let gradient = expected - neuron.forward(&inputs);

            output = neuron.backwards(&inputs, gradient, learning_rate);

            iteration_count += 1;
        }

        println!(
            "converged after {} iterations, with weights: {:?}",
            iteration_count,
            neuron.get_weights_and_bias()
        );
        println!("output: {:?}", output);
    }

    #[test]
    #[should_panic]
    fn test_input_size_mismatch() {
        let neuron = ReluNeuron::new(2);
        let invalid_inputs = vec![1.0, 2.0, 3.0];
        neuron.forward(&invalid_inputs);
    }
}
