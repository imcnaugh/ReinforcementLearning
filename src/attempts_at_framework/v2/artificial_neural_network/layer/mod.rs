mod input_layer;

pub enum LayerType {
    Input,
    Hidden,
    Output,
}

pub trait Layer {
    fn get_type(&self) -> LayerType;
}