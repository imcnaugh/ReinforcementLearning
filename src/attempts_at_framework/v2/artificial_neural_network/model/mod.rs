use crate::attempts_at_framework::v2::artificial_neural_network::layer::Layer;
use crate::attempts_at_framework::v2::artificial_neural_network::neuron::Neuron;

pub struct Model<N>
where
    N: Neuron,
{
    name: String,
    version: String,
    layers: Vec<Box<Layer<N>>>,
}

impl<N> Model<N>
where
    N: Neuron,
{
    pub fn predict(&self, input: Vec<f64>) -> Vec<f64> {
        self.layers
            .iter()
            .fold(input, |acc, layer| layer.forward(&acc))
    }

    pub fn train(&mut self, input: Vec<f64>, expected: Vec<f64>, learning_rate: f64) {
        let predicted = self.predict(input);
        let loss = expected
            .iter()
            .zip(predicted.iter())
            .fold(0.0, |acc, (expected, predicted)| {
                acc + (expected - predicted).powi(2)
            });

        let gradient = expected
            .iter()
            .zip(predicted.iter())
            .map(|(expected, predicted)| expected - predicted)
            .collect::<Vec<f64>>();
        /*
        let mut total_loss = 0.0;
            for (input, expected) in &data {
                let hidden_layer_outputs = hidden_layer.forward(&input[..]);
                let predicted = output_layer.forward(&hidden_layer_outputs[..]);
                let loss = (expected - predicted[0]).powi(2);

                let gradient = expected - predicted[0];
                let gradient_hidden =
                    output_layer.backwards(&hidden_layer_outputs[..], &[gradient], learning_rate);
                let gradient_output =
                    hidden_layer.backwards(&input[..], &gradient_hidden, learning_rate);

                total_loss += loss;
            }
            if epoch % 100 == 0 {
                println!("Epoch: {}, Loss: {}", epoch, total_loss);
            }
         */
    }
}
