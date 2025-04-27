use crate::attempts_at_framework::v2::artificial_neural_network::neuron::linear_neuron::LinearNeuron;
use crate::attempts_at_framework::v2::artificial_neural_network::neuron::relu_neuron::ReluNeuron;
use crate::attempts_at_framework::v2::artificial_neural_network::neuron::Neuron;

pub struct Layer<N: Neuron> {
    neurons: Vec<N>,
    inputs: usize,
}

impl<N> Layer<N>
where
    N: Neuron,
{
    pub fn new_relu_layer(size: usize, inputs: usize) -> Layer<ReluNeuron> {
        let neurons = (0..size)
            .map(|_| ReluNeuron::new(inputs))
            .collect::<Vec<ReluNeuron>>();

        Layer { inputs, neurons }
    }

    pub fn new_linear_layer(size: usize, inputs: usize) -> Layer<LinearNeuron> {
        let neurons = (0..size)
            .map(|_| LinearNeuron::new(inputs))
            .collect::<Vec<LinearNeuron>>();

        Layer { inputs, neurons }
    }

    pub fn calculate(&self, i: &[f64]) -> Vec<f64> {
        assert_eq!(i.len(), self.inputs);
        self.neurons.iter().map(|n| n.forward(i)).collect()
    }

    pub fn learn(&mut self, i: &[f64], expected: &[f64], learning_rate: f64) {
        assert_eq!(i.len(), self.inputs);

        self.neurons.iter_mut().enumerate().for_each(|(index, n)| {
            n.backwards(i, expected[index], learning_rate);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer() {
        // let mut test_layer = Layer::<LinearNeuron>::new_linear_layer(1, 1);
        let mut test_layer = Layer::<ReluNeuron>::new_relu_layer(1, 1);

        let expected = vec![10.0];
        let inputs = vec![1.0];

        let mut current = test_layer.calculate(&inputs);
        let mut error = (current[0] - expected[0]).abs();

        for i in 0..1000 {
            if error < 0.0001 {
                println!("Converged in {} iterations.", i);
                break;
            }
            test_layer.learn(&inputs, &expected, 0.1);
            current = test_layer.calculate(&inputs);
            error = (current[0] - expected[0]).abs();
        }
    }
}
