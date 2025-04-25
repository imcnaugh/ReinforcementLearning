use crate::attempts_at_framework::v2::artificial_neural_network::layer::{Layer, LayerType};

struct InputLayer {
}

impl InputLayer {
    pub fn new() -> Self {
        Self {}
    }
}

impl Layer for InputLayer {
    fn get_type(&self) -> LayerType {
        LayerType::Input
    }
}