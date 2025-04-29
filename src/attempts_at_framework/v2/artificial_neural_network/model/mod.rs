use crate::attempts_at_framework::v2::artificial_neural_network::layer::Layer;
use crate::attempts_at_framework::v2::artificial_neural_network::loss_functions::LossFunction;
use crate::attempts_at_framework::v2::artificial_neural_network::neuron::Neuron;
use crate::chapter_05::policy::Policy;

pub struct Model<N, L>
where
    N: Neuron,
    L: LossFunction,
{
    name: String,
    version: String,
    layers: Vec<Box<Layer<N>>>,
    loss_function: L,
}

impl<N, L> Model<N, L>
where
    N: Neuron,
    L: LossFunction,
{
    pub fn predict(&self, input: Vec<f64>) -> Vec<f64> {
        self.layers
            .iter()
            .fold(input, |acc, layer| layer.forward(&acc))
    }

    pub fn train(&mut self, input: Vec<f64>, expected: Vec<f64>, learning_rate: f64) {
        let predicted = self.predict(input);
        let new_loss = self.loss_function.calculate_loss(&expected, &predicted);

        //TODO verify this is the sam and remove it.
        let _loss = expected
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
