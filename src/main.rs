use eframe::egui;

fn main() {
    let options = eframe::NativeOptions {
        ..Default::default()
    };

    eframe::run_native("My egui App", options, Box::new(|_cc| Ok(Box::new(MyApp))));
}

struct MyApp;

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.label("Hello world!");
        });
    }
}
