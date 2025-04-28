use crate::attempts_at_framework::v2::artificial_neural_network::neuron::linear_neuron::LinearNeuron;
use crate::attempts_at_framework::v2::artificial_neural_network::neuron::relu_neuron::ReluNeuron;
use crate::attempts_at_framework::v2::artificial_neural_network::neuron::Neuron;

struct TwoLayerNetwork {
    hidden_layer: Vec<ReluNeuron>,
    output_layer: LinearNeuron,
}

impl TwoLayerNetwork {
    fn new() -> Self {
        TwoLayerNetwork {
            hidden_layer: vec![ReluNeuron::new(1), ReluNeuron::new(1)],
            output_layer: LinearNeuron::new(2),
        }
    }

    fn forward(&self, x: &[f64]) -> f64 {
        let hidden_outputs = self
            .hidden_layer
            .iter()
            .zip(x)
            .map(|(neuron, &input)| neuron.forward(&[input]))
            .collect::<Vec<f64>>();
        self.output_layer.forward(&hidden_outputs)
    }

    fn train(&mut self, input: &[f64], expected: f64, learning_rate: f64) -> f64 {
        let hidden_outputs: Vec<f64> = self
            .hidden_layer
            .iter()
            .zip(input.iter())
            .map(|(n, &input)| n.forward(&[input]))
            .collect();
        let y_pred = self.output_layer.forward(&hidden_outputs);
        let loss = (expected - y_pred).powi(2);

        let gradient = expected - y_pred;
        let gradient_hidden = self
            .output_layer
            .backwards(&hidden_outputs, gradient, learning_rate);

        for (i, (neuron, &input)) in self.hidden_layer.iter_mut().zip(input).enumerate() {
            neuron.backwards(&[input], gradient_hidden[i], learning_rate);
        }

        loss
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer() {
        let mut network = TwoLayerNetwork::new();
        let learning_rate = 0.0001;

        let data = vec![
            (vec![1.0, 20.0], 21.0),
            (vec![2.0, 30.0], 32.0),
            (vec![3.0, 40.0], 43.0),
        ];

        for epoch in 1..1000 {
            let mut total_loss = 0.0;
            for (input, expected) in &data {
                total_loss += network.train(&input[..], *expected, learning_rate);
            }
            if epoch % 100 == 0 {
                println!("Epoch: {}, Loss: {}", epoch, total_loss);
            }
        }

        println!(
            "Input 4.0 -> Prediction: {}",
            network.forward(&vec![4.0, 90.0])
        ); // Should be ~9.0
    }
}
