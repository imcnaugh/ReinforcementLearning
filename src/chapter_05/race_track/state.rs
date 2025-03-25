pub trait State {
    fn get_id(&self) -> String;
    fn get_actions(&self) -> Vec<String>;
    fn is_terminal(&self) -> bool;
    fn take_action(&self, action: &str) -> (f64, Self);
}
