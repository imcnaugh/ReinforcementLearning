use crate::chapter_10::mountain_car::{
    CarAction, MountainCar, POSITION_LOWER_BOUND, POSITION_UPPER_BOUND, VELOCITY_LOWER_BOUND,
    VELOCITY_UPPER_BOUND,
};
use rand::Rng;

pub fn semi_gradient_sarsa_mountain_car(
    learning_rate: f64,
    discount_factor: f64,
    episodes: usize,
) -> Vec<f64> {
    let mut weights = vec![0.0; 8 * 2 * CarAction::COUNT];

    for episode_number in 0..episodes {
        let starting_x_position = 0.0;
        let mut car = MountainCar::new(starting_x_position, 0.0);
        let mut action = select_action_for_mountain_car(&car, &weights);

        let mut ticks: usize = 0;

        while car.get_x_position() < POSITION_UPPER_BOUND {
            let orig_position = car.get_x_position();
            let orig_velocity = car.get_velocity();
            car.tick(&action);
            let terminal = car.get_x_position() == POSITION_UPPER_BOUND;
            let reward = if terminal { 0.0 } else { -1.0 };

            if terminal {
                let error =
                    reward - state_action_value(orig_position, orig_velocity, &action, &weights);
                update_weights(
                    &mut weights,
                    learning_rate,
                    error,
                    get_values_for_mountain_car(
                        &MountainCar::new(orig_position, orig_velocity),
                        action,
                    ),
                );
            } else {
                let next_action = select_action_for_mountain_car(&car, &weights);

                let new_state_action_value = state_action_value(
                    car.get_x_position(),
                    car.get_velocity(),
                    &next_action,
                    &weights,
                );
                let error = reward + (discount_factor * new_state_action_value)
                    - state_action_value(orig_position, orig_velocity, &action, &weights);
                update_weights(
                    &mut weights,
                    learning_rate,
                    error,
                    get_values_for_mountain_car(&car, next_action),
                );

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

fn state_action_value(x: f64, v: f64, action: &CarAction, weights: &[f64]) -> f64 {
    let values = get_values_for_mountain_car(&MountainCar::new(x, v), *action);
    values.iter().zip(weights).map(|(v, w)| v * w).sum()
}

fn car_action_value(car: &MountainCar, action: &CarAction, weights: &[f64]) -> f64 {
    let values = get_values_for_mountain_car(car, action.clone());
    values.iter().zip(weights).map(|(v, w)| v * w).sum()
}

fn select_action_for_mountain_car(car: &MountainCar, weights: &[f64]) -> CarAction {
    let mut rng = rand::thread_rng();
    if rng.gen::<f64>() < 0.1 {
        let actions = vec![CarAction::Forward, CarAction::Neutral, CarAction::Reverse];
        return actions[rng.gen_range(0..actions.len())];
    }

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
    response[v_index] += 1.0;
    response[x_index] += 1.0;
    response
}

fn get_values_for_mountain_car(car: &MountainCar, action: CarAction) -> Vec<f64> {
    let tiles = 8;

    let v_diff = VELOCITY_UPPER_BOUND - VELOCITY_LOWER_BOUND;
    let v_step = v_diff / (tiles as f64);
    let v_index = ((car.get_velocity() - VELOCITY_LOWER_BOUND) / v_step) as usize;

    let x_diff = POSITION_UPPER_BOUND - POSITION_LOWER_BOUND;
    let x_step = x_diff / (tiles as f64);
    let x_index = ((car.get_x_position() - POSITION_LOWER_BOUND) / x_step) as usize;

    get_values_from_indexes(x_index, v_index, &action)
}

#[cfg(test)]
mod tests {
    use crate::chapter_10::mountain_car::{CarAction, MountainCar};
    use crate::chapter_10::semi_gradient_sarsa_mountian_car::{
        get_values_for_mountain_car, semi_gradient_sarsa_mountain_car,
    };

    #[test]
    fn idk() {
        let car = MountainCar::new(0.0, 0.0);
        let values = get_values_for_mountain_car(&car, CarAction::Reverse);
        assert_eq!(values.len(), 8 * 2 * CarAction::COUNT);
    }

    #[test]
    fn learn() {
        let weights = semi_gradient_sarsa_mountain_car(0.5 / 8.0, 1.0, 100);
    }
}
