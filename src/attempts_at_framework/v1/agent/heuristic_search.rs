use rand::prelude::IteratorRandom;
use simple_chess::chess_game_state_analyzer::GameState;
use simple_chess::{ChessGame, ChessMoveType, Color};
use simple_chess::piece::{ChessPiece, PieceType};

pub fn get_best_action(game: &mut ChessGame, depth: usize) -> String {
    let player_color = game.get_current_players_turn();

    let idk = idk(game, depth, player_color);
    println!("moves and values: {:?}", idk);

    let max_value = idk.iter().map(|x| x.0).fold(f64::MIN, f64::max);
    let best_moves: Vec<_> = idk.iter().filter(|x| x.0 == max_value).cloned().collect();

    best_moves.iter().choose(&mut rand::rng()).unwrap().1.clone()
}

fn idk(game: &mut ChessGame, depth: usize, player_color: Color) -> Vec<(f64, String)> {
    if depth == 0 {
        return vec![(0.0, String::new())];
    }

    let moves = match game.get_game_state() {
        GameState::InProgress { legal_moves, .. } => legal_moves,
        GameState::Check { legal_moves, .. } => legal_moves,
        GameState::Checkmate { .. } => Vec::new(),
        GameState::Stalemate => Vec::new(),
    };

    moves.iter().map(|m| {
        let move_as_string = simple_chess::codec::long_algebraic_notation::encode_move_as_long_algebraic_notation(&m);
        let mut take_reward= match m {
            ChessMoveType::Move { taken_piece, .. } => get_taken_piece_reward(taken_piece),
            ChessMoveType::EnPassant { .. } => 1.0,
            ChessMoveType::Castle { .. } => 0.0,
        };

        if game.get_current_players_turn() != player_color {
            take_reward *= -1.0;
        }

        game.make_move(*m);

        let continuing_reward = match game.get_game_state() {
            GameState::InProgress { .. } => {
                let v = idk(game, depth - 1, player_color);
                let x = if player_color == game.get_current_players_turn() {
                    f64::max
                } else {
                    f64::min
                };
                v.iter().map(|a|a.0).reduce(x).unwrap()
            }
            GameState::Check { .. } => {
                let mut v = idk(game, depth - 1, player_color);
                let x = if player_color == game.get_current_players_turn() {
                    f64::max
                } else {
                    f64::min
                };
                v.iter().map(|a|a.0).reduce(x).unwrap()
            }
            GameState::Checkmate { .. } => {
                1000.0
            },
            GameState::Stalemate => 0.0
        };

        game.undo_last_move();

        let total_reward = take_reward + continuing_reward;
        (total_reward, move_as_string)
    }).collect()
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
