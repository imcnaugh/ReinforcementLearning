use egui::Shape::Vec;
use simple_chess::chess_game_state_analyzer::GameState;
use crate::attempts_at_framework::v1::state::State;

#[derive(Debug, Clone)]
pub struct ChessState {
    id: String,
    moves: Vec<String>,
    is_terminal: bool,
}

impl ChessState {
    pub fn new(game_as_fen_string: String) -> Self {
        let mut game = simple_chess::codec::forsyth_edwards_notation::build_game_from_string(&game_as_fen_string).unwrap();
        let (is_terminal, moves) = match &game.get_game_state() {
                GameState::InProgress { legal_moves, .. } => (false, legal_moves),
                GameState::Check { legal_moves, .. } => (false, legal_moves),
                GameState::Checkmate { .. } => (true, &vec![]),
                GameState::Stalemate => (true, &vec![]),
            };
        let moves = moves.iter().map(|m| simple_chess::codec::long_algebraic_notation::encode_move_as_long_algebraic_notation(m)).collect();

        let regex = regex::Regex::new(r"^(.*) (.) (.*) (.*) (.*) (.*)").unwrap();
        let captures = regex.captures(&game_as_fen_string).unwrap();
        let parts: Vec<String> = captures.iter().skip(1).map(|m| m.unwrap().as_str().to_string()).collect();
        
        let id = format!("{}_{}_{}", parts[0], parts[2], parts[3]);

        Self {
            id,
            moves,
            is_terminal
        }
    }
}

impl State for ChessState {
    fn get_id(&self) -> String {
        self.id.clone()
    }

    fn get_actions(&self) -> Vec<String> {
        self.moves
    }

    fn is_terminal(&self) -> bool {
        self.is_terminal
    }

    fn take_action(&self, action: &str) -> (f64, Self) {
        todo!()
    }
}