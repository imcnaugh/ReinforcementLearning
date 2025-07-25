use crate::attempts_at_framework::v2::state::State;

#[derive(Clone)]
pub struct GridworldState {
    id: String,
    actions: Vec<String>,
    is_terminal: bool,
    values: Vec<f64>,
}

impl GridworldState {}

impl State for GridworldState {
    fn get_id(&self) -> String {
        self.id.clone()
    }

    fn get_actions(&self) -> Vec<String> {
        self.actions.clone()
    }

    fn is_terminal(&self) -> bool {
        self.is_terminal
    }

    fn take_action(&self, action: &str) -> (f64, Self) {
        match self.id.as_str() {
            "left" => match action {
                "left" => (-1.0, generate_left_state()),
                "right" => (-1.0, generate_center_state()),
                _ => panic!("Invalid action"),
            },
            "center" => match action {
                "left" => (-1.0, generate_right_state()),
                "right" => (-1.0, generate_left_state()),
                _ => panic!("Invalid action"),
            },
            "right" => match action {
                "left" => (-1.0, generate_center_state()),
                "right" => (-1.0, generate_terminal_state()),
                _ => panic!("Invalid action"),
            },
            _ => panic!("Invalid id"),
        }
    }

    fn get_values(&self) -> Vec<f64> {
        self.values.clone()
    }
}

pub fn generate_left_state() -> GridworldState {
    GridworldState {
        id: "left".to_string(),
        actions: vec!["l".to_string(), "r".to_string()],
        is_terminal: false,
        values: vec![0.0, 0.0],
    }
}

pub fn generate_center_state() -> GridworldState {
    GridworldState {
        id: "center".to_string(),
        actions: vec!["l".to_string(), "r".to_string()],
        is_terminal: false,
        values: vec![0.0, 0.0],
    }
}

pub fn generate_right_state() -> GridworldState {
    GridworldState {
        id: "right".to_string(),
        actions: vec!["l".to_string(), "r".to_string()],
        is_terminal: false,
        values: vec![0.0, 0.0],
    }
}

pub fn generate_terminal_state() -> GridworldState {
    GridworldState {
        id: "terminal".to_string(),
        actions: Vec::new(),
        is_terminal: true,
        values: vec![0.0, 0.0],
    }
}
