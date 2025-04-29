use crate::attempts_at_framework::v2::artificial_neural_network::neuron::Neuron;

pub struct LinearNeuron {
    weights: Vec<f64>,
    bias: f64,
}

impl LinearNeuron {
    pub fn build(weights: &[f64], bias: f64) -> Result<Self, Box<(dyn std::error::Error)>> {
        let neuron = Self {
            weights: weights.to_vec(),
            bias,
        };
        Ok(neuron)
    }
}

impl Neuron for LinearNeuron {
    fn new(number_of_inputs: usize) -> Self {
        Self {
            weights: vec![0.0; number_of_inputs],
            bias: 0.0,
        }
    }

    fn get_weights_and_bias(&self) -> (&[f64], &f64) {
        (self.weights.as_slice(), &self.bias)
    }

    fn forward(&self, inputs: &[f64]) -> f64 {
        assert_eq!(inputs.len(), self.weights.len());
        inputs
            .iter()
            .zip(self.weights.iter())
            .fold(self.bias, |acc, (x, w)| acc + x * w)
    }

    fn backwards(&mut self, inputs: &[f64], gradient: f64, learning_rate: f64) -> Vec<f64> {
        self.weights
            .iter_mut()
            .zip(inputs)
            .for_each(|(weight, input)| {
                *weight += gradient * input * learning_rate;
            });
        self.bias += gradient * learning_rate;

        self.weights
            .iter()
            .map(|weight| weight * gradient)
            .collect()
    }

    fn activation_derivative(&self, output: f64) -> f64 {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_neuron() {
        let mut neuron = LinearNeuron::new(2);
        let inputs = vec![2.0, 1.0];
        let expected = 2.0;
        let learning_rate = 0.01;

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
}
