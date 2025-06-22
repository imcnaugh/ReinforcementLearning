use crate::chapter_10::mountain_car::{
    feature_vector, CarAction, MountainCar, POSITION_LOWER_BOUND, POSITION_UPPER_BOUND,
    VELOCITY_LOWER_BOUND, VELOCITY_UPPER_BOUND,
};
use rand::prelude::IndexedRandom;
use rand::{rng, Rng};

pub fn semi_gradient_sarsa_mountain_car(
    learning_rate: f64,
    discount_factor: f64,
    episodes: usize,
) -> Vec<f64> {
    let mut weights = vec![0.0; 8 * 2 * CarAction::COUNT];

    let starting_x_positions: Vec<f64> = (0..100)
        .map(|i| {
            let increment = 0.2 / 100.0;
            -0.6 + increment * i as f64
        })
        .collect();

    for episode_number in 0..episodes {
        let starting_x_position = starting_x_positions
            .choose(&mut rand::thread_rng())
            .unwrap();
        let mut car = MountainCar::new(*starting_x_position, 0.0);
        let mut action = select_action_for_mountain_car(&car, &weights);

        let mut ticks: usize = 0;

        while car.get_x_position() < POSITION_UPPER_BOUND {
            let orig_position = car.get_x_position();
            let orig_velocity = car.get_velocity();
            car.tick(&action);
            let terminal = car.get_x_position() == POSITION_UPPER_BOUND;
            let reward = if terminal { 0.0 } else { -1.0 };

            let original_feature_vector = feature_vector(orig_position, orig_velocity, action);
            let original_state_estimate_value =
                state_action_value(&original_feature_vector, &weights);

            if terminal {
                let error = reward - original_state_estimate_value;
                update_weights(&mut weights, learning_rate, error, original_feature_vector);
            } else {
                let next_action = select_action_for_mountain_car(&car, &weights);

                let feature_vector =
                    feature_vector(car.get_x_position(), car.get_velocity(), next_action);
                let next_state_estimated_value = state_action_value(&feature_vector, &weights);

                let error = reward + (discount_factor * next_state_estimated_value)
                    - original_state_estimate_value;
                update_weights(&mut weights, learning_rate, error, original_feature_vector);

                action = next_action;
            }

            ticks += 1;
        }
        println!("Episode {} finished in {} ticks", episode_number, ticks);
    }

    weights
}

fn update_weights(
    weights: &mut Vec<f64>,
    learning_rate: f64,
    error: f64,
    state_action_value: Vec<f64>,
) {
    for (w, v) in weights.iter_mut().zip(state_action_value.iter()) {
        *w += learning_rate * error * *v;
    }
}

fn state_action_value(feature_vector: &[f64], weights: &[f64]) -> f64 {
    feature_vector.iter().zip(weights).map(|(v, w)| v * w).sum()
}

fn car_action_value(car: &MountainCar, action: &CarAction, weights: &[f64]) -> f64 {
    let values =
        get_values_for_mountain_car(car.get_x_position(), car.get_velocity(), action.clone());
    values.iter().zip(weights).map(|(v, w)| v * w).sum()
}

fn select_action_for_mountain_car(car: &MountainCar, weights: &[f64]) -> CarAction {
    let mut rng = rand::thread_rng();
    if rng.gen::<f64>() < 0.1 {
        let actions = vec![CarAction::Forward, CarAction::Neutral, CarAction::Reverse];
        return actions[rng.gen_range(0..actions.len())];
    };

    get_best_action_for_car(car, weights)
}

fn get_best_action_for_car(car: &MountainCar, weights: &[f64]) -> CarAction {
    let actions: Vec<(CarAction, f64)> =
        vec![CarAction::Forward, CarAction::Neutral, CarAction::Reverse]
            .iter()
            .map(|action| (action.clone(), car_action_value(car, action, weights)))
            .collect();

    let mut best_action = actions[0];
    for action in actions {
        if best_action.1 < action.1 {
            best_action = action
        }
    }

    best_action.0
}

fn get_values_from_indexes(x: usize, v: usize, action: &CarAction) -> Vec<f64> {
    let tiles = 8;
    let mut response = vec![0.0; tiles * 2 * CarAction::COUNT];
    let mult = match action {
        CarAction::Forward => 0,
        CarAction::Neutral => 1,
        CarAction::Reverse => 2,
    };
    let buffer = mult * tiles * 2;
    let v_index = buffer + v;
    let x_index = buffer + tiles + x;
    response[v_index] = 1.0;
    response[x_index] = 1.0;
    response
}

fn get_values_for_mountain_car(x_pos: f64, velocity: f64, action: CarAction) -> Vec<f64> {
    let tiles = 8;

    let v_diff = VELOCITY_UPPER_BOUND - VELOCITY_LOWER_BOUND;
    let v_step = v_diff / (tiles as f64);
    let v_index = ((velocity - VELOCITY_LOWER_BOUND) / v_step) as usize;

    let x_diff = POSITION_UPPER_BOUND - POSITION_LOWER_BOUND;
    let x_step = x_diff / (tiles as f64);
    let x_index = ((x_pos - POSITION_LOWER_BOUND) / x_step) as usize;

    get_values_from_indexes(x_index, v_index, &action)
}

#[cfg(test)]
mod tests {
    use crate::chapter_10::mountain_car::{CarAction, MountainCar, POSITION_UPPER_BOUND};
    use crate::chapter_10::semi_gradient_sarsa_mountian_car::{
        get_best_action_for_car, get_values_for_mountain_car, semi_gradient_sarsa_mountain_car,
    };
    use crate::service::{LineChartBuilder, LineChartData};
    use plotters::prelude::full_palette::GREEN_900;
    use plotters::prelude::{ShapeStyle, GREEN};
    use plotters::style::full_palette::{BLUE_500, RED_500, YELLOW_500};
    use plotters::style::{RED, YELLOW};
    use std::path::PathBuf;

    #[test]
    fn idk() {
        let car = MountainCar::new(0.0, 0.0);
        let values = get_values_for_mountain_car(
            car.get_x_position(),
            car.get_velocity(),
            CarAction::Reverse,
        );
        assert_eq!(values.len(), 8 * 2 * CarAction::COUNT);
    }

    #[test]
    fn learn() {
        let weights = semi_gradient_sarsa_mountain_car(0.5 / 8.0, 1.0, 100);

        let mut test_car = MountainCar::new(-0.6, 0.0);
        let mut x_pos_and_action: Vec<(f64, CarAction)> = vec![];

        for i in 0..500 {
            if test_car.get_x_position() == POSITION_UPPER_BOUND {
                break;
            }
            let action = get_best_action_for_car(&test_car, &weights);
            x_pos_and_action.push((test_car.get_x_position(), action));
            test_car.tick(&action);
        }

        let mut chart_builder = LineChartBuilder::new();

        let mut previous_action = x_pos_and_action[0].1;
        let mut relevant_data: Vec<(f32, f32)> = vec![];

        for (index, (x_pos, action)) in x_pos_and_action.iter().enumerate() {
            if *action != previous_action {
                let color = match previous_action {
                    CarAction::Forward => ShapeStyle::from(&GREEN_900),
                    CarAction::Neutral => ShapeStyle::from(&BLUE_500),
                    CarAction::Reverse => ShapeStyle::from(&RED_500),
                };
                chart_builder.add_data(LineChartData::new_with_style(
                    "".to_string(),
                    relevant_data,
                    color,
                ));
                relevant_data = vec![];
            }
            relevant_data.push((index as f32, *x_pos as f32));
            previous_action = *action;
        }

        let color = match previous_action {
            CarAction::Forward => ShapeStyle::from(&GREEN_900),
            CarAction::Neutral => ShapeStyle::from(&BLUE_500),
            CarAction::Reverse => ShapeStyle::from(&RED_500),
        };
        chart_builder.add_data(LineChartData::new_with_style(
            "".to_string(),
            relevant_data,
            color,
        ));

        chart_builder.set_path(PathBuf::from("output/chapter10/trained_mountain_car.png"));

        chart_builder.create_chart().unwrap();
    }
}
