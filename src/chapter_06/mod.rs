pub mod one_step_temporal_difference;
pub mod blackjack_test_state;

#[cfg(test)]
mod tests{
    use std::collections::HashMap;
    use std::path::PathBuf;
    use plotters::prelude::{ShapeStyle, BLACK, BLUE};
    use crate::chapter_05::race_track::state::State;
    use rand::prelude::IndexedRandom;
    use crate::service::{LineChartBuilder, LineChartData};

    #[test]
    fn example_6_2_markov_reward_process(){
        let mut rng = rand::rng();
        let discount_rate = 1.0;
        let size_step_parameter = 0.1;
        let number_of_episodes = 100;
        let mut state_values: HashMap<String, f64> = HashMap::new();

        (0..5).for_each(|i| {
            state_values.insert(i.to_string(), 0.5);
        });
        state_values.insert(String::from("6"), 0.0);

        (0..number_of_episodes).for_each(|i| {
            let mut state = MrpState::new(2, false);

            while !state.is_terminal() {
                let actions = [String::from("l"), String::from("r")];
                let action = actions.choose(&mut rng).unwrap();
                let (reward, new_state) = state.take_action(action);
                let current_state_value = state_values.get(&state.get_id()).unwrap();
                let new_state_value = state_values.get(&new_state.get_id()).unwrap();
                let error = reward + (discount_rate * new_state_value) - current_state_value;
                let new_state_value = current_state_value + (size_step_parameter * error);
                state_values.insert(state.get_id(), new_state_value);
                state = new_state;
            }
        });

        let data_after_100_episodes = (0..5).map(|i| {
            let value = state_values.get(&i.to_string()).unwrap();
            (i as f32, *value as f32)
        }).collect::<Vec<(f32, f32)>>();

        let actual_value = (0..5).map(|i| {
            let value:f32 = (i as f32 + 1.0) / 6.0;
            (i as f32, value)
        }).collect();

        let data_all_episodes = LineChartData::new(
            format!("after {} episodes", number_of_episodes),
            data_after_100_episodes,
            ShapeStyle::from(&BLUE),
        );

        let data_expected = LineChartData::new(
            String::from("expected value"),
            actual_value,
            ShapeStyle::from(&BLACK),
        );

        let mut builder = LineChartBuilder::new();
        builder
            .set_path(PathBuf::from("output/chapter6/mrp.png"))
            .add_data(data_expected)
            .add_data(data_all_episodes);
        builder.create_chart().unwrap();
    }

    struct MrpState {
        id: u8,
        is_terminal: bool,
    }

    impl MrpState {
        fn new(id: u8, is_terminal: bool) -> Self {
            Self {
                id,
                is_terminal,
            }
        }
    }

    impl State for MrpState {
        fn get_id(&self) -> String {
            self.id.to_string()
        }

        fn get_actions(&self) -> Vec<String> {
           vec![String::from("l"), String::from("r")]
        }

        fn is_terminal(&self) -> bool {
            self.is_terminal
        }

        fn take_action(&self, action: &str) -> (f64, Self) {
            if self.id == 0 && action == "l" {
                return (0.0, MrpState::new(6, true));
            }
            if self.id == 4 && action == "r" {
                return (1.0, MrpState::new(6, true));
            }

            let new_id = match action {
                "l" => self.id - 1,
                "r" => self.id + 1,
                _ => panic!()
            };
            (0.0, MrpState::new(new_id, false))
        }
    }
}