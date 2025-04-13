use simple_chess::chess_game_state_analyzer::GameState;
use simple_chess::{ChessGame, ChessMoveType, Color};
use simple_chess::piece::{ChessPiece, PieceType};

pub fn get_best_action(game: ChessGame, depth: usize) -> String {
    let mut game = game.clone();
    let moves = match game.get_game_state() {
        GameState::InProgress { legal_moves, .. } => legal_moves,
        GameState::Check { legal_moves, .. } => legal_moves,
        GameState::Checkmate { .. } => Vec::new(),
        GameState::Stalemate => Vec::new(),
    };

    let idk = moves.iter().map(|&m| {
        let move_as_string = simple_chess::codec::long_algebraic_notation::encode_move_as_long_algebraic_notation(&m);
        let mut interim_game = game.clone();
        interim_game.make_move(m);
        let new_state = interim_game.get_game_state();
        let value: f64 = match new_state {
            GameState::InProgress { legal_moves, .. } => {
                get_average_value_of_possible_moves(depth, game.clone(), legal_moves)
            }
            GameState::Check { legal_moves, .. } => {
                get_average_value_of_possible_moves(depth, game.clone(), legal_moves)
            }
            GameState::Checkmate { .. } => 1000.0,
            GameState::Stalemate => 0.0,
        };

        (move_as_string, value)
    }).max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();

    idk.0.clone()
}

fn get_average_value_of_possible_moves(depth: usize, mut game: ChessGame, legal_moves: Vec<ChessMoveType>) -> f64 {
    let s = legal_moves.iter().map(|nm| {
        let mut interim_game = game.clone();
        interim_game.make_move(*nm);
        idk(interim_game, depth, game.get_current_players_turn().opposite())
    }).sum::<f64>();
    s / legal_moves.len() as f64
}

fn idk(game: ChessGame, depth: usize, player_color: Color) -> f64 {
    if depth == 0 {
        return 0.0;
    }
    let players_turn = game.get_current_players_turn() == player_color;

    let next_depth = match players_turn {
        true => depth - 1,
        false => depth
    };

    let mut game = game.clone();
    let moves = match game.get_game_state() {
        GameState::InProgress { legal_moves, .. } => legal_moves,
        GameState::Check { legal_moves, .. } => legal_moves,
        GameState::Checkmate { .. } => Vec::new(),
        GameState::Stalemate => Vec::new(),
    };

    moves.iter().map(|m| {
        let mut interim_game = game.clone();
        interim_game.make_move(*m);
        let mut base_reward = 0.0;
        match m {
            ChessMoveType::Move { taken_piece, .. } => {
                base_reward += get_taken_piece_reward(players_turn, taken_piece);
            }
            ChessMoveType::EnPassant { taken_piece, .. } => {
                base_reward += get_taken_piece_reward(players_turn, &Some(*taken_piece));
            }
            ChessMoveType::Castle { .. } => {}
        }

        let continuing_reward = match interim_game.get_game_state() {
            GameState::InProgress { legal_moves, .. } => {
                let s = legal_moves.iter().map(|nm| {
                    let mut interim_game = game.clone();
                    interim_game.make_move(*nm);
                    idk(interim_game, next_depth, player_color)
                }).sum::<f64>();
                s / legal_moves.len() as f64
            }
            GameState::Check { legal_moves, .. } => {
                let s = legal_moves.iter().map(|nm| {
                    let mut interim_game = game.clone();
                    interim_game.make_move(*nm);
                    idk(interim_game, next_depth, player_color)
                }).sum::<f64>();
                s / legal_moves.len() as f64
            }
            GameState::Checkmate { winner } => {
                if players_turn {
                    1000.0
                } else {
                    -1000.0
                }
            },
            GameState::Stalemate => 0.0
        };

        base_reward + continuing_reward
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
