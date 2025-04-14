use std::cmp::Ordering;
use simple_chess::chess_game_state_analyzer::GameState;
use simple_chess::{ChessGame, ChessMoveType, Color};
use simple_chess::codec::forsyth_edwards_notation::encode_game_as_string;
use simple_chess::piece::{ChessPiece, PieceType};

pub fn get_best_action(game: &mut ChessGame, depth: usize) -> String {
    let moves = match game.get_game_state() {
        GameState::InProgress { legal_moves, .. } => legal_moves,
        GameState::Check { legal_moves, .. } => legal_moves,
        GameState::Checkmate { .. } => Vec::new(),
        GameState::Stalemate => Vec::new(),
    };

    let idk = moves.iter().map(|&m| {
        let move_as_string = simple_chess::codec::long_algebraic_notation::encode_move_as_long_algebraic_notation(&m);
        game.make_move(m);
        let value: f64 = match game.get_game_state() {
            GameState::InProgress { legal_moves, .. } => {
                get_average_value_of_possible_moves(depth, game, legal_moves)
            }
            GameState::Check { legal_moves, .. } => {
                get_average_value_of_possible_moves(depth, game, legal_moves)
            }
            GameState::Checkmate { .. } => 1000.0,
            GameState::Stalemate => 0.0,
        };

        game.undo_last_move();

        (move_as_string, value)
    }).max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal)).unwrap();

    idk.0.clone()
}

fn get_average_value_of_possible_moves(depth: usize, game: &mut ChessGame, legal_moves: Vec<ChessMoveType>) -> f64 {
    if depth == 0 {
        return 0.0;
    }

    let board_as_fen_string = encode_game_as_string(&game);
    // println!("finding best action for board state: {} and depth: {}", board_as_fen_string, depth);
    let s = legal_moves.iter().map(|nm| {
        let move_as_string = simple_chess::codec::long_algebraic_notation::encode_move_as_long_algebraic_notation(&nm);
        game.make_move(*nm);
        let value = idk(game, depth, game.get_current_players_turn().opposite());
        game.undo_last_move();
        value
    }).sum::<f64>();
    s / legal_moves.len() as f64
}

fn idk(game: &mut ChessGame, depth: usize, player_color: Color) -> f64 {
    let players_turn = game.get_current_players_turn() == player_color;

    let moves = match game.get_game_state() {
        GameState::InProgress { legal_moves, .. } => legal_moves,
        GameState::Check { legal_moves, .. } => legal_moves,
        GameState::Checkmate { .. } => Vec::new(),
        GameState::Stalemate => Vec::new(),
    };

    moves.iter().map(|m| {
        game.make_move(*m);
        let take_reward= match m {
            ChessMoveType::Move { taken_piece, .. } => get_taken_piece_reward(players_turn, taken_piece),
            ChessMoveType::EnPassant { .. } => 1.0,
            ChessMoveType::Castle { .. } => 0.0,
        };

        let continuing_reward = match game.get_game_state() {
            GameState::InProgress { legal_moves, .. } => {
                get_average_value_of_possible_moves(depth - 1, game, legal_moves)
            }
            GameState::Check { legal_moves, .. } => {
                get_average_value_of_possible_moves(depth - 1, game, legal_moves)
            }
            GameState::Checkmate { .. } => {
                if players_turn {
                    1000.0
                } else {
                    -1000.0
                }
            },
            GameState::Stalemate => 0.0
        };
        game.undo_last_move();
        take_reward + continuing_reward
    }).sum::<f64>() / moves.len() as f64
}

fn get_taken_piece_reward(players_turn: bool, taken_piece: &Option<ChessPiece>) -> f64 {
    if let Some(taken_piece) = taken_piece {
        let take_reward = match taken_piece.get_piece_type() {
            PieceType::Pawn => 1.0,
            PieceType::Rook => 5.0,
            PieceType::Knight => 3.0,
            PieceType::Bishop => 3.0,
            PieceType::Queen => 9.0,
            PieceType::King => 0.0,
        };

        if players_turn {
            take_reward * -1.0
        } else {
            take_reward
        }
    } else {
        0.0
    }
}
