use crate::attempts_at_framework::v2::artificial_neural_network::layer::Layer;
use crate::attempts_at_framework::v2::artificial_neural_network::loss_functions::LossFunction;
use crate::attempts_at_framework::v2::artificial_neural_network::neuron::Neuron;

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
        let predicted = self.predict(input.clone());
        let new_loss = self.loss_function.calculate_loss(&expected, &predicted);

        //TODO verify this is the sam and remove it.
        let _loss = expected
            .iter()
            .zip(predicted.iter())
            .fold(0.0, |acc, (expected, predicted)| {
                acc + (expected - predicted).powi(2)
            });

        let mut gradient = self
            .loss_function
            .calculate_gradient(&expected, &predicted)
            .iter()
            .zip(predicted.iter())
            .zip(self.layers.last().unwrap().get_neurons().iter())
            .map(|((loss_gradient, predicted), neuron)| {
                loss_gradient * neuron.activation_derivative(*predicted)
            })
            .collect::<Vec<f64>>();
        let mut current_inputs = input;

        self.layers.iter_mut().rev().for_each(|layer| {
            // TODO might need to add to total loss, figure that out
            current_inputs = layer.forward(&current_inputs);
            gradient = layer.backwards(&current_inputs, &gradient, learning_rate);
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::attempts_at_framework::v2::artificial_neural_network::neuron::linear_neuron::LinearNeuron;
    use crate::attempts_at_framework::v2::artificial_neural_network::neuron::relu_neuron::ReluNeuron;

    #[test]
    fn test_model() {
        let output_layer = Layer::<LinearNeuron>::new(1, 2);
        let hidden_layer = Layer::<ReluNeuron>::new(2, 2);

        let boxed_output_layer = Box::new(output_layer);
        let boxed_hidden_layer = Box::new(hidden_layer);

        let layers = vec![boxed_hidden_layer, boxed_output_layer];
    }
}
