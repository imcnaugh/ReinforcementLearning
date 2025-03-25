use eframe::egui;
use egui::TextBuffer;
use std::path::Path;
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

    let path = Path::new("resources/tracks/simple_racetrack.txt");
    let track = parse_track_from_file(path).unwrap();

    eframe::run_native(
        "My egui App",
        options,
        Box::new(|_cc| {
            let mut app = MyApp::new();
            app.set_track(track);
            app.do_learning();
            Ok(Box::new(app))
        }),
    )
    .unwrap();
}

struct MyApp {
    track: Option<RaceTrack>,
    sim_starting_x: String,
    sim_starting_y: String,
    target_policy: Option<DeterministicPolicy>,
}

impl MyApp {
    pub fn new() -> Self {
        Self {
            track: None,
            sim_starting_x: String::from("0"),
            sim_starting_y: String::from("0"),
            target_policy: None,
        }
    }

    pub fn set_track(&mut self, track: RaceTrack) {
        self.track = Some(track);
    }

    pub fn do_learning(&mut self) {
        if self.track.is_none() {
            panic!("No track set");
        }
        let track = self.track.as_ref().unwrap();

        let starting_states: Vec<Racer> = track
            .get_start_positions()
            .iter()
            .map(|starting_position| {
                let starting_position = (starting_position.0 as i32, starting_position.1 as i32);
                Racer::new(starting_position, track)
            })
            .collect();

        let mut learner = MonteCarloOffPolicyLearner::new(starting_states, 1.0);

        learner.learn_for_episodes(1000000);

        self.target_policy = Some(learner.get_target_policy().clone());
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let mut track_string = match &self.track {
            None => String::new(),
            Some(t) => t.to_string(),
        };

        if ctx.input(|i| i.key_pressed(egui::Key::ArrowUp)) {
            if let Ok(value) = self.sim_starting_y.parse::<i32>() {
                self.sim_starting_y = (value - 1).to_string();
            }
        }
        if ctx.input(|i| i.key_pressed(egui::Key::ArrowDown)) {
            if let Ok(value) = self.sim_starting_y.parse::<i32>() {
                self.sim_starting_y = (value + 1).to_string();
            }
        }
        if ctx.input(|i| i.key_pressed(egui::Key::ArrowLeft)) {
            if let Ok(value) = self.sim_starting_x.parse::<i32>() {
                self.sim_starting_x = (value - 1).to_string();
            }
        }
        if ctx.input(|i| i.key_pressed(egui::Key::ArrowRight)) {
            if let Ok(value) = self.sim_starting_x.parse::<i32>() {
                self.sim_starting_x = (value + 1).to_string();
            }
        }

        let player_position = (
            self.sim_starting_x.parse::<i32>().unwrap_or(0),
            self.sim_starting_y.parse::<i32>().unwrap_or(0),
        );

        let action_by_state = self
            .target_policy
            .as_ref()
            .unwrap()
            .pick_action_for_state(&format!("{}_{}_0_0", player_position.0, player_position.1))
            .unwrap_or("no");

        track_string = track_string
            .lines()
            .enumerate()
            .map(|(y, line)| {
                line.chars()
                    .enumerate()
                    .map(|(x, ch)| {
                        if (x as i32, y as i32) == player_position {
                            '^'
                        } else {
                            ch
                        }
                    })
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n");

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.label(egui::RichText::new("Hello world!").monospace());
            ui.label(egui::RichText::new(&track_string).monospace());
            ui.horizontal(|ui| {
                ui.label("Starting X:");
                ui.text_edit_singleline(&mut self.sim_starting_x)
            });
            ui.horizontal(|ui| {
                ui.label("Starting Y:");
                ui.text_edit_singleline(&mut self.sim_starting_y);
            });
            ui.label(egui::RichText::new(&format!("Action: {}", action_by_state)).monospace());
        });
    }
}
