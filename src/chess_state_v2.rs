use crate::attempts_at_framework::v2::state::State;
use rand::prelude::IndexedRandom;
use simple_chess::chess_game_state_analyzer::GameState;
use simple_chess::game_board::Board;
use simple_chess::piece::ChessPiece;

#[derive(Debug, Clone)]
pub struct ChessStateV2 {
    id: String,
    fen_string: String,
    moves: Vec<String>,
    is_terminal: bool,
    board: Board<ChessPiece>,
}

pub fn get_state_id_from_fen_string(game_as_fen_string: &String) -> String {
    let regex = regex::Regex::new(r"^(.*) (.) (.*) (.*) (.*) (.*)").unwrap();
    let captures = regex.captures(&game_as_fen_string).unwrap();
    let parts: Vec<String> = captures
        .iter()
        .skip(1)
        .map(|m| m.unwrap().as_str().to_string())
        .collect();

    format!("{}_{}_{}", parts[0], parts[2], parts[3])
}

impl ChessStateV2 {
    pub fn new(game_as_fen_string: String) -> Self {
        let mut game = simple_chess::codec::forsyth_edwards_notation::build_game_from_string(
            &game_as_fen_string,
        )
        .unwrap();
        let (is_terminal, moves) = match game.get_game_state() {
            GameState::InProgress { legal_moves, .. } => {
                let can_be_over = game.can_claim_draw().is_some();

                (can_be_over, legal_moves)
            }
            GameState::Check { legal_moves, .. } => {
                let can_be_over = game.can_claim_draw().is_some();

                (can_be_over, legal_moves)
            }
            GameState::Checkmate { .. } => (true, vec![]),
            GameState::Stalemate => (true, vec![]),
        };
        let moves = moves.iter().map(|m| simple_chess::codec::long_algebraic_notation::encode_move_as_long_algebraic_notation(m)).collect();

        let id = crate::chess_state::get_state_id_from_fen_string(&game_as_fen_string);

        Self {
            id,
            fen_string: game_as_fen_string,
            moves,
            is_terminal,
            board: game.get_board().clone(),
        }
    }
}

impl State for ChessStateV2 {
    fn get_id(&self) -> String {
        self.id.clone()
    }

    fn get_actions(&self) -> Vec<String> {
        self.moves.clone()
    }

    fn is_terminal(&self) -> bool {
        self.is_terminal
    }

    fn take_action(&self, action: &str) -> (f64, Self) {
        let mut rng = rand::rng();
        let mut game =
            simple_chess::codec::forsyth_edwards_notation::build_game_from_string(&self.fen_string)
                .unwrap();
        let possible_moves = match game.get_game_state() {
            GameState::InProgress { legal_moves, .. } => legal_moves,
            GameState::Check { legal_moves, .. } => legal_moves,
            GameState::Checkmate { .. } => Vec::new(),
            GameState::Stalemate => Vec::new(),
        };
        let move_to_take = possible_moves
            .iter()
            .find(|m|
                simple_chess::codec::long_algebraic_notation::encode_move_as_long_algebraic_notation(m) == action)
            .unwrap();
        game.make_move(move_to_take.clone());
        match game.get_game_state() {
            GameState::InProgress { .. } => {
                let next_possible_moves = match game.get_game_state() {
                    GameState::InProgress { legal_moves, .. } => legal_moves,
                    GameState::Check { legal_moves, .. } => legal_moves,
                    GameState::Checkmate { .. } => Vec::new(),
                    GameState::Stalemate => Vec::new(),
                };
                let next_move = next_possible_moves.choose(&mut rng).unwrap();
                game.make_move(next_move.clone());
                let new_fen_string =
                    simple_chess::codec::forsyth_edwards_notation::encode_game_as_string(&game);
                match game.get_game_state() {
                    GameState::InProgress { .. } => (0.0, ChessStateV2::new(new_fen_string)),
                    GameState::Check { .. } => (0.0, ChessStateV2::new(new_fen_string)),
                    GameState::Checkmate { .. } => (-1.0, ChessStateV2::new(new_fen_string)),
                    GameState::Stalemate => (0.0, ChessStateV2::new(new_fen_string)),
                }
            }
            GameState::Check { .. } => {
                let next_possible_moves = match game.get_game_state() {
                    GameState::InProgress { legal_moves, .. } => legal_moves,
                    GameState::Check { legal_moves, .. } => legal_moves,
                    GameState::Checkmate { .. } => Vec::new(),
                    GameState::Stalemate => Vec::new(),
                };
                let next_move = next_possible_moves.choose(&mut rng).unwrap();
                game.make_move(next_move.clone());
                let new_fen_string =
                    simple_chess::codec::forsyth_edwards_notation::encode_game_as_string(&game);
                match game.get_game_state() {
                    GameState::InProgress { .. } => (0.0, ChessStateV2::new(new_fen_string)),
                    GameState::Check { .. } => (0.0, ChessStateV2::new(new_fen_string)),
                    GameState::Checkmate { .. } => (-1.0, ChessStateV2::new(new_fen_string)),
                    GameState::Stalemate => (0.0, ChessStateV2::new(new_fen_string)),
                }
            }
            GameState::Checkmate { .. } => {
                let fen_string =
                    simple_chess::codec::forsyth_edwards_notation::encode_game_as_string(&game);
                (1.0, ChessStateV2::new(fen_string))
            }
            GameState::Stalemate => {
                let fen_string =
                    simple_chess::codec::forsyth_edwards_notation::encode_game_as_string(&game);
                (0.0, ChessStateV2::new(fen_string))
            }
        }
    }

    fn get_values(&self) -> Vec<f64> {
        let mut values = Vec::new();
        values.push(1.0);

        let width = self.board.get_width();
        let height = self.board.get_height();

        for x in 0..height {
            for y in 0..width {
                match self.board.get_piece_at_space(x, y) {
                    None => values.push(0.0),
                    Some(p) => {}
                }
            }
        }

        values
    }
}
