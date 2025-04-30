use crate::attempts_at_framework::v2::artificial_neural_network::layer::Layer;
use crate::attempts_at_framework::v2::artificial_neural_network::loss_functions::LossFunction;
use crate::attempts_at_framework::v2::artificial_neural_network::model::Model;
use crate::attempts_at_framework::v2::artificial_neural_network::neuron::linear_neuron::LinearNeuron;
use crate::attempts_at_framework::v2::artificial_neural_network::neuron::relu_neuron::ReluNeuron;

pub struct ModelBuilder {
    name: Option<String>,
    version: Option<String>,
    layers: Vec<LayerBuilder>,
    input_size: Option<usize>,
    loss_function: Option<Box<dyn LossFunction>>,
}

pub enum LayerType {
    RELU,
    LINEAR,
}

pub struct LayerBuilder {
    layer_type: LayerType,
    number_of_neurons: usize,
}

impl LayerBuilder {
    pub fn new(layer_type: LayerType, number_of_neurons: usize) -> Self {
        Self {
            layer_type,
            number_of_neurons,
        }
    }

    pub fn build(&self, number_of_inputs: usize) -> Layer {
        match self.layer_type {
            LayerType::RELU => Layer::new::<ReluNeuron>(self.number_of_neurons, number_of_inputs),
            LayerType::LINEAR => {
                Layer::new::<LinearNeuron>(self.number_of_neurons, number_of_inputs)
            }
        }
    }
}

impl ModelBuilder {
    pub fn new() -> Self {
        Self {
            name: None,
            version: None,
            layers: Vec::new(),
            input_size: None,
            loss_function: None,
        }
    }

    pub fn set_name(&mut self, name: String) -> &mut Self {
        self.name = Some(name);
        self
    }

    pub fn set_version(&mut self, version: String) -> &mut Self {
        self.version = Some(version);
        self
    }

    pub fn set_input_size(&mut self, input_size: usize) -> &mut Self {
        self.input_size = Some(input_size);
        self
    }

    pub fn set_loss_function(&mut self, loss_function: Box<dyn LossFunction>) -> &mut Self {
        self.loss_function = Some(loss_function);
        self
    }

    pub fn add_layer(&mut self, layer_builder: LayerBuilder) -> &mut Self {
        self.layers.push(layer_builder);
        self
    }

    pub fn clear_layers(&mut self) -> &mut Self {
        self.layers.clear();
        self
    }

    pub fn build(mut self) -> Result<Model, Box<dyn std::error::Error>> {
        if self.loss_function.is_none() {
            return Err("Loss function is not set")?;
        }
        if self.input_size.is_none() {
            return Err("Input size is not set")?;
        }
        if self.layers.len() == 0 {
            return Err("No layers have been added")?;
        }

        let mut next_input_size = self.input_size.unwrap();
        let layers: Vec<Box<Layer>> = self
            .layers
            .iter()
            .map(|layer_builder| {
                let layer = layer_builder.build(next_input_size);
                next_input_size = layer_builder.number_of_neurons;
                Box::new(layer)
            })
            .collect();

        let model = Model::new(
            self.name.clone().unwrap_or_else(|| "Unnamed".to_string()),
            self.version.clone().unwrap_or_else(|| "0.0.1".to_string()),
            layers,
            self.loss_function.unwrap(),
        );
        Ok(model)
    }
}

#[cfg(test)]
mod tests {
    use crate::attempts_at_framework::v2::artificial_neural_network::loss_functions::mean_squared_error::MeanSquaredError;
    use super::*;

    #[test]
    fn test_builder() {
        let mut builder = ModelBuilder::new();
        builder
            .set_name("test model".to_string())
            .set_version("1.0".to_string())
            .set_loss_function(Box::new(MeanSquaredError))
            .set_input_size(2);

        builder.add_layer(LayerBuilder::new(LayerType::RELU, 2));
        builder.add_layer(LayerBuilder::new(LayerType::LINEAR, 1));

        let mut model = builder.build().unwrap();

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

        let test_date = vec![50.0, 2.0];
        let test_expected = 102.0;

        let predicted = model.predict(test_date);
        let loss = model
            .get_loss_function()
            .calculate_loss(&vec![test_expected], &predicted);

        println!("Loss: {}", loss);
        println!("Predicted: {:?}", predicted);
    }
}
