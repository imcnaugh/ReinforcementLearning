use eframe::egui;
use egui::Ui;
use simple_chess::chess_game_state_analyzer::GameState;
use simple_chess::codec::long_algebraic_notation::encode_move_as_long_algebraic_notation;
use simple_chess::{ChessGame, ChessMoveType};

fn main() {
    let options = eframe::NativeOptions {
        ..Default::default()
    };

    eframe::run_native(
        "My egui App",
        options,
        Box::new(|_cc| {
            let app = MyApp::new();
            Ok(Box::new(app))
        }),
    )
    .unwrap();
}

struct MyApp {
    chess_game: ChessGame,
    selected_square: Option<(usize, usize)>,
    game_state: GameState,
    possible_moves: Vec<(usize, usize, ChessMoveType)>,
    previous_moves: Vec<ChessMoveType>,
}

impl MyApp {
    pub fn new() -> Self {
        let mut game = ChessGame::new();
        let state = game.get_game_state();

        Self {
            chess_game: game,
            game_state: state,
            selected_square: None,
            possible_moves: Vec::new(),
            previous_moves: Vec::new(),
        }
    }

    pub fn do_learning(&mut self) {
        // let mut learner = MonteCarloOffPolicyLearner::new(starting_states, 1.0);
        //
        // learner.learn_for_episodes(1000000);
    }

    fn square_selected(&mut self, row: usize, col: usize) {
        match self.selected_square {
            None => {
                self.select_piece_to_move(row, col);
            }
            Some((selected_row, selected_col)) => {
                if selected_row == row && selected_col == col {
                    self.selected_square = None;
                    self.possible_moves = Vec::new();
                    return;
                }

                let complete_move = self
                    .possible_moves
                    .iter()
                    .find(|&&(s_x, s_y, _)| s_x == row && s_y == col);

                match complete_move {
                    None => self.select_piece_to_move(row, col),
                    Some((_, _, m)) => {
                        self.previous_moves.push(m.clone());
                        self.chess_game.make_move(*m);
                        self.selected_square = None;
                        self.possible_moves = Vec::new();
                        self.game_state = self.chess_game.get_game_state();
                    }
                }
            }
        }
    }

    fn select_piece_to_move(&mut self, row: usize, col: usize) {
        self.selected_square = Some((row, col));
        let moves: Vec<(usize, usize, ChessMoveType)> = match &self.game_state {
            GameState::InProgress { legal_moves, turn } => {
                Self::select_from_legal_moves(row, col, legal_moves)
            }
            GameState::Check { legal_moves, .. } => {
                Self::select_from_legal_moves(row, col, legal_moves)
            }
            GameState::Checkmate { .. } => {
                vec![]
            }
            GameState::Stalemate => {
                vec![]
            }
        };
        self.possible_moves = moves;
    }

    fn select_from_legal_moves(
        row: usize,
        col: usize,
        legal_moves: &Vec<ChessMoveType>,
    ) -> Vec<(usize, usize, ChessMoveType)> {
        let moves_from_here = legal_moves
            .iter()
            .filter(|&m| match m {
                ChessMoveType::Move {
                    original_position, ..
                } => original_position.0 == row && original_position.1 == col,
                ChessMoveType::EnPassant {
                    original_position, ..
                } => original_position.0 == row && original_position.1 == col,
                ChessMoveType::Castle {
                    king_original_position,
                    ..
                } => king_original_position.0 == row && king_original_position.1 == col,
            })
            .collect::<Vec<&ChessMoveType>>();

        let can_move_to = moves_from_here
            .iter()
            .map(|&m| match m {
                ChessMoveType::Move { new_position, .. } => {
                    (new_position.0, new_position.1, m.clone())
                }
                ChessMoveType::EnPassant { new_position, .. } => {
                    (new_position.0, new_position.1, m.clone())
                }
                ChessMoveType::Castle {
                    king_new_position, ..
                } => (king_new_position.0, king_new_position.1, m.clone()),
            })
            .collect::<Vec<(usize, usize, ChessMoveType)>>();
        can_move_to
    }

    fn draw_chess_board(&mut self, ui: &mut Ui) {
        ui.vertical_centered(|ui| {
            ui.spacing_mut().item_spacing = egui::vec2(0.0, 0.0);
            (0..self.chess_game.get_board().get_height())
                .rev()
                .for_each(|y| {
                    ui.horizontal(|ui| {
                        ui.add_sized(
                            [40.0, 40.0],
                            egui::Label::new(egui::RichText::new((y + 1).to_string()).size(30.0)),
                        );
                        ui.spacing_mut().item_spacing = egui::vec2(0.0, 0.0);
                        (0..self.chess_game.get_board().get_width()).for_each(|x| {
                            let piece = self.chess_game.get_board().get_piece_at_space(x, y);
                            let square_color = if (x + y) % 2 == 0 {
                                egui::Color32::from_rgb(138, 144, 145)
                            } else {
                                egui::Color32::from_rgb(181, 181, 181)
                            };
                            let piece_as_char = match piece {
                                None => " ",
                                Some(p) => p.as_utf_str(),
                            };

                            let is_selected = if let Some((row, col)) = self.selected_square {
                                row == x && col == y
                            } else {
                                false
                            };

                            let can_move_to = self
                                .possible_moves
                                .iter()
                                .any(|&(s_x, s_y, _)| s_x == x && s_y == y);

                            if ui
                                .add_sized(
                                    [40.0, 40.0],
                                    egui::Button::new(
                                        egui::RichText::new(piece_as_char)
                                            .monospace()
                                            .color(egui::Color32::from_rgb(0, 0, 0))
                                            .size(30.0),
                                    )
                                    .frame(false)
                                    .corner_radius(0.0)
                                    .fill(square_color)
                                    .stroke(if is_selected {
                                        egui::Stroke::new(1.5, egui::Color32::from_rgb(0, 0, 255))
                                    } else if can_move_to {
                                        egui::Stroke::new(1.5, egui::Color32::from_rgb(0, 255, 0))
                                    } else {
                                        egui::Stroke::NONE
                                    }),
                                )
                                .clicked()
                            {
                                self.square_selected(x, y);
                            };
                        });
                    });
                });
            ui.horizontal(|ui| {
                ui.add_sized([40.0, 40.0], egui::Label::new(""));
                ('A'..='H').for_each(|x| {
                    ui.add_sized(
                        [40.0, 40.0],
                        egui::Label::new(egui::RichText::new(x).size(30.0)),
                    );
                });
            });
        });
    }

    fn reset_game_button(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            if ui.button("New Game").clicked() {
                self.chess_game = ChessGame::new();
                self.game_state = self.chess_game.get_game_state();
                self.selected_square = None;
                self.possible_moves = Vec::new();
                self.previous_moves = Vec::new();
            };
        });
    }

    fn previous_moves(&mut self, ui: &mut Ui) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.vertical(|ui| {
                self.previous_moves
                    .chunks(2)
                    .enumerate()
                    .for_each(|(turn_number, m)| {
                        ui.horizontal(|ui| {
                            ui.label(format!("{}.", turn_number + 1));
                            m.iter().for_each(|c| {
                                let text = encode_move_as_long_algebraic_notation(c);
                                ui.label(text);
                                ui.add_space(10.0);
                            });
                        });
                    });
            });
        });
    }

    fn game_status_label(&mut self, ui: &mut Ui) {
        let current_game_state = match &self.game_state {
            GameState::InProgress { turn, .. } => {
                format!("Turn: {:?}", turn)
            }
            GameState::Check { turn, .. } => {
                format!("Check! Turn: {:?}", turn)
            }
            GameState::Checkmate { winner } => {
                format!("Checkmate! Winner: {:?}", winner)
            }
            GameState::Stalemate => String::from("Stalemate"),
        };
        ui.label(current_game_state);
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                self.draw_chess_board(ui);
                ui.add_space(10.0);
                ui.vertical(|ui| {
                    self.game_status_label(ui);
                    ui.add_space(10.0);
                    self.reset_game_button(ui);
                    ui.add_space(10.0);
                    self.previous_moves(ui);
                });
            });
        });
    }
}
