use crate::attempts_at_framework::v2::artificial_neural_network::neuron::linear_neuron::LinearNeuron;
use crate::attempts_at_framework::v2::artificial_neural_network::neuron::relu_neuron::ReluNeuron;
use crate::attempts_at_framework::v2::artificial_neural_network::neuron::Neuron;
use std::sync::atomic::AtomicUsize;

static LAYER_COUNT: AtomicUsize = AtomicUsize::new(0);

pub struct Layer<N>
where
    N: Neuron,
{
    id: usize,
    name: Option<String>,
    neurons: Vec<N>,
    input_count: usize,
}

impl<N: Neuron> Layer<N> {
    pub fn new(neuron_count: usize, input_count: usize) -> Self
    where
        N: Neuron + 'static,
    {
        let neurons = (0..neuron_count).map(|_| N::new(input_count)).collect();
        let id = LAYER_COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Layer {
            id,
            name: None,
            neurons,
            input_count,
        }
    }

    pub fn get_id(&self) -> usize {
        self.id
    }

    pub fn get_neurons(&self) -> &Vec<N> {
        &self.neurons
    }

    pub fn get_name(&self) -> Option<&String> {
        self.name.as_ref()
    }

    pub fn forward(&self, inputs: &[f64]) -> Vec<f64> {
        self.neurons.iter().map(|n| n.forward(inputs)).collect()
    }

    pub fn backwards(&mut self, inputs: &[f64], gradient: &[f64], learning_rate: f64) -> Vec<f64> {
        self.neurons
            .iter_mut()
            .zip(gradient)
            .map(|(neuron, gradient)| neuron.backwards(inputs, *gradient, learning_rate))
            .fold(vec![0.0; self.input_count], |acc, x| {
                acc.iter().zip(x.iter()).map(|(&a, &b)| a + b).collect()
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_layer_structs() {
        let learning_rate = 0.01;
        let data = vec![
            (vec![2.0, 1.0], 5.0),
            (vec![1.0, 2.0], 4.0),
            (vec![3.0, 3.0], 9.0),
        ];
        let test_date = vec![4.0, 2.0];
        let test_expected = 10.0;

        let mut hidden_layer = Layer::<ReluNeuron>::new(2, 2);
        let mut output_layer = Layer::<LinearNeuron>::new(1, 2);

        for epoch in 1..1000 {
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
        }

        let hidden_layer_outputs = hidden_layer.forward(&test_date[..]);
        let predicted = output_layer.forward(&hidden_layer_outputs[..]);
        println!("Input 4.0, 2.0 -> Prediction: {}", predicted[0]);
    }
}
