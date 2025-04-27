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

    fn forward(&self, x: f64) -> f64 {
        let hidden_outputs = self
            .hidden_layer
            .iter()
            .map(|neuron| neuron.forward(&[x]))
            .collect::<Vec<f64>>();
        self.output_layer.forward(&hidden_outputs)
    }

    fn train(&mut self, x: f64, y_true: f64, learning_rate: f64) -> f64 {
        let hidden_outputs: Vec<f64> = self.hidden_layer.iter().map(|n| n.forward(&[x])).collect();
        let y_pred = self.output_layer.forward(&hidden_outputs);
        let loss = (y_true - y_pred).powi(2);

        let gradient = y_pred - y_true;
        let gradient_hidden = self
            .output_layer
            .backwards(&hidden_outputs, gradient, learning_rate);

        for (i, neuron) in self.hidden_layer.iter_mut().enumerate() {
            neuron.backwards(&[x], gradient_hidden[i], learning_rate);
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
        let learning_rate = 0.01;

        let data = vec![(1.0, 3.0), (2.0, 5.0), (3.0, 7.0)];

        for epoch in 1..1000 {
            let mut total_loss = 0.0;
            for &(x, y_true) in &data {
                total_loss += network.train(x, y_true, learning_rate);
            }
            if epoch % 100 == 0 {
                println!("Epoch: {}, Loss: {}", epoch, total_loss);
            }
        }

        println!("Input 4.0 -> Prediction: {}", network.forward(4.0)); // Should be ~9.0
    }
}
