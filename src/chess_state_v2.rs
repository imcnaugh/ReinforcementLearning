use crate::attempts_at_framework::v2::state::State;
use simple_chess::chess_game_state_analyzer::GameState;
use simple_chess::game_board::{get_column_and_row_from_square_name, Board};
use simple_chess::piece::{ChessPiece, PieceType};
use simple_chess::{ChessGame, ChessMoveType, Color};

#[derive(Clone)]
pub struct ChessStateV2 {
    id: String,
    fen_string: String,
    moves: Vec<String>,
    is_terminal: bool,
    board: Board<ChessPiece>,
    select_other_player_moves_fn: fn(&mut ChessGame) -> ChessMoveType,
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
    pub fn new(
        game_as_fen_string: String,
        select_other_player_moves_fn: fn(&mut ChessGame) -> ChessMoveType,
    ) -> Self {
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
            select_other_player_moves_fn,
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
                let next_move = (self.select_other_player_moves_fn)(&mut game);
                game.make_move(next_move);
                let new_fen_string =
                    simple_chess::codec::forsyth_edwards_notation::encode_game_as_string(&game);
                match game.get_game_state() {
                    GameState::InProgress { .. } => (
                        0.0,
                        ChessStateV2::new(new_fen_string, self.select_other_player_moves_fn),
                    ),
                    GameState::Check { .. } => (
                        0.0,
                        ChessStateV2::new(new_fen_string, self.select_other_player_moves_fn),
                    ),
                    GameState::Checkmate { .. } => (
                        -1.0,
                        ChessStateV2::new(new_fen_string, self.select_other_player_moves_fn),
                    ),
                    GameState::Stalemate => (
                        0.0,
                        ChessStateV2::new(new_fen_string, self.select_other_player_moves_fn),
                    ),
                }
            }
            GameState::Check { .. } => {
                let next_move = (self.select_other_player_moves_fn)(&mut game);
                game.make_move(next_move);
                let new_fen_string =
                    simple_chess::codec::forsyth_edwards_notation::encode_game_as_string(&game);
                match game.get_game_state() {
                    GameState::InProgress { .. } => (
                        0.0,
                        ChessStateV2::new(new_fen_string, self.select_other_player_moves_fn),
                    ),
                    GameState::Check { .. } => (
                        0.0,
                        ChessStateV2::new(new_fen_string, self.select_other_player_moves_fn),
                    ),
                    GameState::Checkmate { .. } => (
                        -1.0,
                        ChessStateV2::new(new_fen_string, self.select_other_player_moves_fn),
                    ),
                    GameState::Stalemate => (
                        0.0,
                        ChessStateV2::new(new_fen_string, self.select_other_player_moves_fn),
                    ),
                }
            }
            GameState::Checkmate { .. } => {
                let fen_string =
                    simple_chess::codec::forsyth_edwards_notation::encode_game_as_string(&game);
                (
                    1.0,
                    ChessStateV2::new(fen_string, self.select_other_player_moves_fn),
                )
            }
            GameState::Stalemate => {
                let fen_string =
                    simple_chess::codec::forsyth_edwards_notation::encode_game_as_string(&game);
                (
                    0.0,
                    ChessStateV2::new(fen_string, self.select_other_player_moves_fn),
                )
            }
        }
    }

    fn get_values(&self) -> Vec<f64> {
        let regex = regex::Regex::new(r"^(.*) (.) (.*) (.*) (.*) (.*)").unwrap();
        let captures = regex.captures(&self.fen_string).unwrap();
        let parts: Vec<String> = captures
            .iter()
            .skip(1)
            .map(|m| m.unwrap().as_str().to_string())
            .collect();

        let width = self.board.get_width();
        let height = self.board.get_height();
        let square_count = width * height;

        let mut black_pawn_map = vec![0.0; square_count];
        let mut white_pawn_map = vec![0.0; square_count];
        let mut black_knight_map = vec![0.0; square_count];
        let mut white_knight_map = vec![0.0; square_count];
        let mut black_bishop_map = vec![0.0; square_count];
        let mut white_bishop_map = vec![0.0; square_count];
        let mut black_rook_map = vec![0.0; square_count];
        let mut white_rook_map = vec![0.0; square_count];
        let mut black_queen_map = vec![0.0; square_count];
        let mut white_queen_map = vec![0.0; square_count];
        let mut black_king_map = vec![0.0; square_count];
        let mut white_king_map = vec![0.0; square_count];

        for x in 0..height {
            for y in 0..width {
                if let Some(piece) = self.board.get_piece_at_space(x, y) {
                    let (b_map, w_map) = match piece.get_piece_type() {
                        PieceType::Pawn => (&mut black_pawn_map, &mut white_pawn_map),
                        PieceType::Rook => (&mut black_rook_map, &mut white_rook_map),
                        PieceType::Knight => (&mut black_knight_map, &mut white_knight_map),
                        PieceType::Bishop => (&mut black_bishop_map, &mut white_bishop_map),
                        PieceType::Queen => (&mut black_queen_map, &mut white_queen_map),
                        PieceType::King => (&mut black_king_map, &mut white_king_map),
                    };
                    let index = x * width + y;
                    match piece.get_color() {
                        Color::White => {
                            w_map[index] = 1.0;
                        }
                        Color::Black => {
                            b_map[index] = 1.0;
                        }
                    }
                }
            }
        }

        let mut castling_rights_map = vec![0.0; 4];
        if parts[2].contains("K") {
            castling_rights_map[0] = 1.0;
        }
        if parts[3].contains("Q") {
            castling_rights_map[1] = 1.0;
        }
        if parts[3].contains("k") {
            castling_rights_map[2] = 1.0;
        }
        if parts[3].contains("q") {
            castling_rights_map[3] = 1.0;
        }

        let mut en_passant_squares_map = vec![0.0; 16];
        if parts[3] != "-" {
            let (col, row) = get_column_and_row_from_square_name(&parts[3]).unwrap();
            let buffer = if row == 5 { 8 } else { 0 };
            en_passant_squares_map[col + buffer] = 1.0;
        }

        let player_turn_map = match parts[1].as_str() {
            "w" => vec![1.0, 0.0],
            "b" => vec![0.0, 1.0],
            _ => vec![0.0, 0.0],
        };

        let mut values = Vec::new();
        values.extend(black_pawn_map); // 64
        values.extend(white_pawn_map); // 64
        values.extend(black_knight_map); // 64
        values.extend(white_knight_map); // 64
        values.extend(black_bishop_map); // 64
        values.extend(white_bishop_map); // 64
        values.extend(black_rook_map); // 64
        values.extend(white_rook_map); // 64
        values.extend(black_queen_map); // 64
        values.extend(white_queen_map); // 64
        values.extend(black_king_map); // 64
        values.extend(white_king_map); // 64
        values.extend(castling_rights_map); // 4
        values.extend(en_passant_squares_map); // 16
        values.extend(player_turn_map); // 2
        values
    }
}
