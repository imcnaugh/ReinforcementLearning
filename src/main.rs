use eframe::egui;
use egui::{TextBuffer, Ui};
use simple_chess::piece::ChessPiece;
use simple_chess::ChessGame;
use std::path::Path;
use simple_chess::chess_game_state_analyzer::GameState;
use ReinforcementLearning::chapter_05::policy::{DeterministicPolicy, Policy};
use ReinforcementLearning::chapter_05::race_track::learning::MonteCarloOffPolicyLearner;
use ReinforcementLearning::chapter_05::race_track::racer::Racer;
use ReinforcementLearning::chapter_05::race_track::state::State;
use ReinforcementLearning::chapter_05::race_track::track::RaceTrack;
use ReinforcementLearning::chapter_05::race_track::track_parser::parse_track_from_file;

fn main() {
    let options = eframe::NativeOptions {
        ..Default::default()
    };

    eframe::run_native(
        "My egui App",
        options,
        Box::new(|_cc| {
            let mut app = MyApp::new();
            Ok(Box::new(app))
        }),
    )
    .unwrap();
}

struct MyApp {
    chess_game: ChessGame,
    selected_square: Option<(usize, usize)>,
}

impl MyApp {
    pub fn new() -> Self {
        Self {
            chess_game: ChessGame::new(),
            selected_square: None,
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
                self.selected_square = Some((row, col));
                match self.chess_game.get_game_state() {
                    GameState::InProgress { legal_moves, turn } => {

                    }
                    GameState::Check { legal_moves, turn } => {}
                    GameState::Checkmate { winner } => {}
                    GameState::Stalemate => {}
                }
            },
            Some(_) => {}
        }
        println!("Square selected: ({}, {})", row, col);
    }

    fn draw_chess_board(&mut self, ui: &mut Ui) {
        ui.vertical_centered(|ui| {
            ui.spacing_mut().item_spacing = egui::vec2(0.0, 0.0);
            (0..self.chess_game.get_board().get_height()).rev().for_each(|y| {
                ui.horizontal(|ui| {
                    ui.add_sized(
                        [40.0, 40.0],
                        egui::Label::new(egui::RichText::new((y + 1).to_string()).size(30.0)),
                    );
                    ui.spacing_mut().item_spacing = egui::vec2(0.0, 0.0);
                    (0..self.chess_game.get_board().get_width()).for_each(|x| {
                        let piece = self.chess_game.get_board().get_piece_at_space(x, y);
                        let square_color = if (x + y) % 2 == 0 {
                            egui::Color32::from_rgb(181, 181, 181)
                        } else {
                            egui::Color32::from_rgb(138, 144, 145)
                        };
                        let piece_as_char = match piece {
                            None => " ",
                            Some(p) => p.as_utf_str(),
                        };
                        if ui.add_sized(
                            [40.0, 40.0],
                            egui::Button::new(
                                egui::RichText::new(piece_as_char)
                                    .monospace()
                                    .color(egui::Color32::from_rgb(0, 0, 0))
                                    .size(30.0),
                            )
                                .frame(false)
                                .corner_radius(0.0)
                                .fill(square_color),
                        ).clicked() {
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
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            self.draw_chess_board(ui);
        });
    }
}
