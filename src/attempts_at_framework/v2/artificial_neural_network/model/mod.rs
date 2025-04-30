use crate::attempts_at_framework::v2::artificial_neural_network::layer::Layer;
use crate::attempts_at_framework::v2::artificial_neural_network::loss_functions::LossFunction;
use crate::attempts_at_framework::v2::artificial_neural_network::neuron::Neuron;

pub struct Model<L>
where
    L: LossFunction,
{
    name: String,
    version: String,
    layers: Vec<Box<Layer>>,
    loss_function: L,
}

impl<L> Model<L>
where
    L: LossFunction,
{
    pub fn predict(&self, input: Vec<f64>) -> Vec<f64> {
        self.layers
            .iter()
            .fold(input, |acc, layer| layer.forward(&acc))
    }

    pub fn train(&mut self, input: Vec<f64>, expected: Vec<f64>, learning_rate: f64) -> f64 {
        let mut layer_inputs = self
            .layers
            .iter()
            .fold(vec![input.clone()], |mut inputs, layer| {
                let predicted = layer.forward(&inputs.last().unwrap());
                inputs.push(predicted);
                inputs
            });

        // removing the last element as that is from the output layer and is the prediction
        let prediction = layer_inputs.pop().unwrap();

        let loss = self.loss_function.calculate_loss(&expected, &prediction);

        /*
        Hot dam ok this needs an explanation, this logic finds the gradient of the output layer

        But due to the back propagation process, each layer starting with the output layer needs
        the gradient calculated via the chain rule from the previous layer; this is the response
        from the `backwards` function provided by each layer. So we update this with the previous
        layers gradients, but it needs to be primed for the output layer.
         */
        let mut gradient = self
            .loss_function
            .calculate_gradient(&expected, &prediction)
            .iter()
            .zip(prediction.iter())
            .zip(self.layers.last().unwrap().get_neurons().iter())
            .map(|((loss_gradient, predicted), neuron)| {
                loss_gradient * neuron.activation_derivative(*predicted)
            })
            .collect::<Vec<f64>>();

        self.layers
            .iter_mut()
            .rev()
            .zip(layer_inputs.iter().rev())
            .for_each(|(layer, inputs)| {
                // TODO might need to add to total loss, figure that out
                gradient = layer.backwards(&inputs, &gradient, learning_rate);
            });

        loss
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::attempts_at_framework::v2::artificial_neural_network::neuron::linear_neuron::LinearNeuron;
    use crate::attempts_at_framework::v2::artificial_neural_network::neuron::relu_neuron::ReluNeuron;
    use crate::attempts_at_framework::v2::artificial_neural_network::loss_functions::mean_squared_error::MeanSquaredError;

    #[test]
    fn test_model() {
        let output_layer = Layer::new::<LinearNeuron>(1, 2);
        let hidden_layer = Layer::new::<ReluNeuron>(2, 2);

        let boxed_output_layer = Box::new(output_layer);
        let boxed_hidden_layer = Box::new(hidden_layer);

        let layers = vec![boxed_hidden_layer, boxed_output_layer];

        let mut model = Model {
            name: "test".to_string(),
            version: "1.0".to_string(),
            layers,
            loss_function: MeanSquaredError,
        };

        let learning_rate = 0.01;
        let data = vec![
            (vec![2.0, 1.0], 5.0),
            (vec![1.0, 2.0], 4.0),
            (vec![3.0, 3.0], 9.0),
        ];

        for epoch in 1..1000 {
            let total_loss: f64 /* Type */ = data.iter().map(|(input, expected)| {
                model.train(input.clone(), vec![expected.clone()], learning_rate)
            }).sum();

            if epoch % 100 == 0 {
                println!("Epoch: {} Loss: {}", epoch, total_loss);
            }
        }

        let test_date = vec![4.0, 2.0];
        let test_expected = 10.0;

        let predicted = model.predict(test_date.clone());
        let loss = model
            .loss_function
            .calculate_loss(&vec![test_expected], &predicted);

        println!("Loss: {}", loss);
        println!("Predicted: {:?}", predicted);
    }
}
