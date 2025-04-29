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
        unimplemented!()
    }
}
