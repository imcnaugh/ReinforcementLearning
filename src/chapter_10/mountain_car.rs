use crate::attempts_at_framework::v2::state::State;
use std::fmt::Display;

pub const VELOCITY_LOWER_BOUND: f64 = -0.07;
pub const VELOCITY_UPPER_BOUND: f64 = 0.07;
pub const POSITION_LOWER_BOUND: f64 = -1.2;
pub const POSITION_UPPER_BOUND: f64 = 0.5;

pub fn feature_vector(x_position: f64, velocity: f64, action: CarAction) -> Vec<f64> {
    const TILES: usize = 8;

    let velocity_tile = get_velocity_tile(velocity, TILES);
    let position_tile = TILES + get_position_tile(x_position, TILES);

    let action_buffer = match action {
        CarAction::Forward => 0,
        CarAction::Neutral => TILES * 2,
        CarAction::Reverse => 2 * TILES * 2,
    };

    let mut response = vec![0.0; TILES * 2 * CarAction::COUNT];
    response[action_buffer + velocity_tile] = 1.0;
    response[action_buffer + position_tile] = 1.0;
    response
}

fn get_position_tile(x_position: f64, tiles: usize) -> usize {
    let position_tile_size: f64 = (POSITION_UPPER_BOUND - POSITION_LOWER_BOUND) / tiles as f64;
    ((x_position - POSITION_LOWER_BOUND) / position_tile_size) as usize
}

fn get_velocity_tile(velocity: f64, tiles: usize) -> usize {
    let velocity_tile_size: f64 = (VELOCITY_UPPER_BOUND - VELOCITY_LOWER_BOUND) / tiles as f64;
    ((velocity - VELOCITY_LOWER_BOUND) / velocity_tile_size) as usize
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CarAction {
    Forward,
    Neutral,
    Reverse,
}

impl CarAction {
    pub fn get_acceleration(&self) -> f64 {
        match self {
            CarAction::Forward => 1.0,
            CarAction::Neutral => 0.0,
            CarAction::Reverse => -1.0,
        }
    }

    pub const COUNT: usize = 3;
}

impl Display for CarAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CarAction::Forward => write!(f, "forward"),
            CarAction::Neutral => write!(f, "neutral"),
            CarAction::Reverse => write!(f, "reverse"),
        }
    }
}

#[derive(Clone)]
pub struct MountainCar {
    x_position: f64,
    velocity: f64,
}

impl MountainCar {
    pub fn new(x_position: f64, velocity: f64) -> Self {
        Self {
            x_position: x_position.clamp(POSITION_LOWER_BOUND, POSITION_UPPER_BOUND),
            velocity: velocity.clamp(VELOCITY_LOWER_BOUND, VELOCITY_UPPER_BOUND),
        }
    }

    pub fn tick(&mut self, action: &CarAction) {
        let new_velocity = ((self.velocity + (0.001 * action.get_acceleration()))
            - (0.0025 * f64::cos(3.0 * self.x_position)))
        .clamp(VELOCITY_LOWER_BOUND, VELOCITY_UPPER_BOUND);

        let new_x_position =
            (self.x_position + new_velocity).clamp(POSITION_LOWER_BOUND, POSITION_UPPER_BOUND);

        self.velocity = new_velocity;
        self.x_position = new_x_position;
    }

    pub fn get_x_position(&self) -> f64 {
        self.x_position
    }

    pub fn get_velocity(&self) -> f64 {
        self.velocity
    }
}

impl State for MountainCar {
    fn get_id(&self) -> String {
        let x_position_tile = get_position_tile(self.get_x_position(), 8);
        let velocity_tile = get_velocity_tile(self.get_velocity(), 8);
        format!("{}_{}", x_position_tile, velocity_tile)
    }

    fn get_actions(&self) -> Vec<String> {
        vec![CarAction::Forward, CarAction::Neutral, CarAction::Reverse]
            .iter()
            .map(|action| action.to_string())
            .collect()
    }

    fn is_terminal(&self) -> bool {
        self.get_x_position() >= POSITION_UPPER_BOUND
    }

    fn take_action(&self, action: &str) -> (f64, Self) {
        let parsed_action = match action {
            "forward" => CarAction::Forward,
            "neutral" => CarAction::Neutral,
            "reverse" => CarAction::Reverse,
            _ => panic!("invalid action"),
        };
        let mut new_state = self.clone();
        new_state.tick(&parsed_action);

        let terminal = new_state.is_terminal();
        let reward = if terminal { 0.0 } else { -1.0 };
        (reward, new_state)
    }

    fn get_values(&self) -> Vec<f64> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::service::{LineChartBuilder, LineChartData};
    use plotters::prelude::full_palette::PURPLE;
    use plotters::prelude::{ShapeStyle, BLACK, BLUE, RED};
    use std::path::PathBuf;

    #[test]
    fn test_mountain_car() {
        let mut go_car = MountainCar {
            x_position: 0.0,
            velocity: 0.0,
        };
        let mut neutral_car = MountainCar {
            x_position: 0.0,
            velocity: 0.0,
        };
        let mut reverse_car = MountainCar {
            x_position: 0.0,
            velocity: 0.0,
        };

        let mut go_car_x_over_time: Vec<(f32, f32)> = Vec::new();
        let mut neutral_car_x_over_time: Vec<(f32, f32)> = Vec::new();
        let mut reverse_car_x_over_time: Vec<(f32, f32)> = Vec::new();

        for tick in 0..100 {
            go_car_x_over_time.push((tick as f32, go_car.x_position as f32));
            neutral_car_x_over_time.push((tick as f32, neutral_car.x_position as f32));
            reverse_car_x_over_time.push((tick as f32, reverse_car.x_position as f32));
            go_car.tick(&CarAction::Forward);
            neutral_car.tick(&CarAction::Neutral);
            reverse_car.tick(&CarAction::Reverse);
        }

        let go_car_data = LineChartData::new_with_style(
            "always accelerate".to_string(),
            go_car_x_over_time,
            ShapeStyle::from(&BLUE),
        );
        let neutral_car_data = LineChartData::new_with_style(
            "always neutral".to_string(),
            neutral_car_x_over_time,
            ShapeStyle::from(&PURPLE),
        );
        let reverse_car_data = LineChartData::new_with_style(
            "always reverse".to_string(),
            reverse_car_x_over_time,
            ShapeStyle::from(&RED),
        );
        let track_width = LineChartData::new_with_style(
            "track width".to_string(),
            vec![
                (0.0, POSITION_LOWER_BOUND as f32),
                (0.0, POSITION_UPPER_BOUND as f32),
            ],
            ShapeStyle::from(&BLACK),
        );

        let mut chart_builder = LineChartBuilder::new();
        chart_builder
            .set_title("car x pos over time".to_string())
            .set_x_label("time".to_string())
            .set_y_label("x position".to_string())
            .set_path(PathBuf::from("output/chapter10/mountain_car_x_pos.png"))
            .add_data(go_car_data)
            .add_data(neutral_car_data)
            .add_data(reverse_car_data)
            .add_data(track_width);

        chart_builder.create_chart().unwrap();
    }
}
