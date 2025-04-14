use std::cmp::Ordering;
use rand::prelude::IteratorRandom;
use simple_chess::chess_game_state_analyzer::GameState;
use simple_chess::{ChessGame, ChessMoveType, Color};
use simple_chess::codec::forsyth_edwards_notation::encode_game_as_string;
use simple_chess::piece::{ChessPiece, PieceType};

pub fn get_best_action(game: &mut ChessGame, depth: usize) -> String {
    let player_color = game.get_current_players_turn();
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
            GameState::InProgress { .. } => {
                idk(game, depth, player_color)
            }
            GameState::Check { .. } => {
                idk(game, depth, player_color)
            }
            GameState::Checkmate { .. } => 1000.0,
            GameState::Stalemate => 0.0,
        };

        game.undo_last_move();

        (move_as_string, value)
    }).collect::<Vec<(String, f64)>>();


    let max_value = idk.iter().map(|x| x.1).fold(f64::MIN, f64::max);
    let best_moves: Vec<_> = idk.iter().filter(|x| x.1 == max_value).cloned().collect();
    println!("Best moves: {:?}", best_moves);

    best_moves.iter().choose(&mut rand::rng()).unwrap().0.clone()
}

fn idk(game: &mut ChessGame, depth: usize, player_color: Color) -> f64 {
    if depth == 0 {
        return 0.0;
    }

    let players_turn = game.get_current_players_turn() == player_color;

    let moves = match game.get_game_state() {
        GameState::InProgress { legal_moves, .. } => legal_moves,
        GameState::Check { legal_moves, .. } => legal_moves,
        GameState::Checkmate { .. } => Vec::new(),
        GameState::Stalemate => Vec::new(),
    };

    moves.iter().map(|m| {
        let mut take_reward= match m {
            ChessMoveType::Move { taken_piece, .. } => get_taken_piece_reward(taken_piece),
            ChessMoveType::EnPassant { .. } => 1.0,
            ChessMoveType::Castle { .. } => 0.0,
        };

        game.make_move(*m);

        let mut continuing_reward = match game.get_game_state() {
            GameState::InProgress { .. } => {
                idk(game, depth - 1, player_color)
            }
            GameState::Check { .. } => {
                idk(game, depth - 1, player_color)
            }
            GameState::Checkmate { .. } => {
                1000.0
            },
            GameState::Stalemate => 0.0
        };

        game.undo_last_move();
        take_reward + continuing_reward
    }).sum::<f64>() / moves.len() as f64
}

fn get_taken_piece_reward(taken_piece: &Option<ChessPiece>) -> f64 {
    if let Some(taken_piece) = taken_piece {
        match taken_piece.get_piece_type() {
            PieceType::Pawn => 1.0,
            PieceType::Rook => 5.0,
            PieceType::Knight => 3.0,
            PieceType::Bishop => 3.0,
            PieceType::Queen => 9.0,
            _ => 0.0,
        }
    } else {
        0.0
    }
}
