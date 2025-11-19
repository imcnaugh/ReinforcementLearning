use crate::attempts_at_framework::v2::artificial_neural_network::neuron::Neuron;

pub struct SigmoidNeuron {
    weights: Vec<f64>,
    bias: f64,
}

impl SigmoidNeuron {
    pub fn build(weights: &[f64], bias: f64) -> Result<Self, Box<(dyn std::error::Error)>> {
        let neuron = Self {
            weights: weights.to_vec(),
            bias,
        };
        Ok(neuron)
    }
}

impl Neuron for SigmoidNeuron {
    fn new(number_of_inputs: usize) -> Self
    where
        Self: Sized,
    {
        Self {
            weights: vec![1.0; number_of_inputs],
            bias: 0.0,
        }
    }

    fn get_weights_and_bias(&self) -> (&[f64], &f64) {
        (self.weights.as_slice(), &self.bias)
    }

    fn forward(&self, inputs: &[f64]) -> f64 {
        let sum = inputs
            .iter()
            .zip(self.weights.iter())
            .fold(self.bias, |acc, (x, w)| acc + x * w);
        1.0 / (1.0 + (-sum).exp())
    }

    fn backwards(&mut self, inputs: &[f64], gradient: f64, learning_rate: f64) -> Vec<f64> {
        let current_value = self.forward(inputs);

        todo!()
    }

    fn activation_derivative(&self, output: f64) -> f64 {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_activation() {
        let neuron = SigmoidNeuron::new(1);
        let inputs = vec![0.0];
        let expected = 0.5;
        assert_eq!(neuron.forward(&inputs), expected);
    }
}
