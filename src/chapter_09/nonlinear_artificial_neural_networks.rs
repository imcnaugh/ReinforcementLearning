use std::f64::consts::E;

struct SigmoidNeuron {
    number_of_inputs: usize,
    weights: Vec<f64>,
    bias: f64,
}

impl SigmoidNeuron {
    fn new(number_of_inputs: usize) -> Self {
        SigmoidNeuron {
            number_of_inputs,
            weights: vec![0.0; number_of_inputs],
            bias: 0.0,
        }
    }

    fn activate(&self, inputs: &[f64]) -> f64 {
        assert_eq!(self.number_of_inputs, inputs.len());

        let weighted_inputs: f64 = inputs
            .iter()
            .zip(self.weights.iter())
            .map(|(&x, &w)| x * w)
            .sum::<f64>()
            + self.bias;

        1.0 / (E.powf(-1.0 * weighted_inputs) + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid_neuron() {
        let mut neuron = SigmoidNeuron::new(2);
        neuron.weights[0] = 0.3;
        neuron.weights[1] = -0.2;
        neuron.bias = 0.1;

        let output = neuron.activate(&[0.5, 0.8]);
        println!("Output: {}", output);
        let diff = (output - 0.5224848247918001).abs();
        println!("Diff: {}", diff);
        assert!(diff < 0.0000000000000001);
    }
}
