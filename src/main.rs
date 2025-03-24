use eframe::egui;
use std::path::Path;
use ReinforcementLearning::chapter_05::race_track::model::RaceTrack;
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
            Ok(Box::new(app))
        }),
    )
    .unwrap();
}

struct MyApp {
    track: Option<RaceTrack>,
}

impl MyApp {
    pub fn new() -> Self {
        Self { track: None }
    }

    pub fn set_track(&mut self, track: RaceTrack) {
        self.track = Some(track);
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let track_string = match &self.track {
            None => String::new(),
            Some(t) => t.to_string(),
        };

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.label(egui::RichText::new("Hello world!").monospace());
            ui.label(egui::RichText::new(&track_string).monospace());
        });
    }
}
