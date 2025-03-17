pub mod blackjack;
pub mod cards;
mod policy;
mod state;

#[cfg(test)]
mod tests {
    use crate::chapter_05::blackjack::State;
    use crate::chapter_05::cards::RandomCardProvider;
    use crate::service::MultiLineChartBuilder;
    use crate::service::MultiLineChartData;
    use rand::Rng;
    use std::collections::HashMap;
    use std::path::PathBuf;

    fn hit_unless_above_20(state: &mut State<RandomCardProvider>) {
        loop {
            let current_count = state.get_player_count();
            let usable_ace = state.get_usable_ace();

            if current_count > 19 {
                break;
            }
            state.hit();
        }
    }

    #[test]
    fn plot_blackjack_state_value() {
        let mut multi_line_cart_builder = MultiLineChartBuilder::new();
        let card_provider: RandomCardProvider = RandomCardProvider::new();

        (2..=11).for_each(|dealer_showing| {
            let mut average_rewards: Vec<f64> = vec![];
            (12..=21).for_each(|player_count| {
                let player_start_count = player_count;
                let player_usable_aces = true;
                let dealer_showing_start = dealer_showing;

                let mut running_average: f64 = 0.00;

                (0..500000).for_each(|i| {
                    let mut state = State::new(
                        player_start_count,
                        dealer_showing_start,
                        player_usable_aces,
                        &card_provider,
                    );
                    hit_unless_above_20(&mut state);
                    let reward = state.check_for_win();
                    running_average = crate::service::calc_average(running_average, i + 1, reward);
                });

                average_rewards.push(running_average);
            });
            let mut multi_line_chart_data = MultiLineChartData::new(average_rewards);
            multi_line_chart_data.set_label(format!("Dealer shows {}", dealer_showing));
            multi_line_cart_builder.add_data(multi_line_chart_data);
        });

        multi_line_cart_builder
            .set_path(PathBuf::from(
                "output/chapter5/blackJack_values_static_policy.png",
            ))
            .set_title(format!("Blackjack Value"));
        multi_line_cart_builder.create_chart().unwrap();
    }

    fn get_state_action_id(state_id: &str, action: bool) -> String {
        format!("{}_{}", state_id, action)
    }

    pub fn get_state_id(player_count: &u8, dealer_showing: &u8, usable_ace: &bool) -> String {
        format!("{}_{}_{}", player_count, dealer_showing, usable_ace)
    }

    fn play_episode(
        policy: &HashMap<String, bool>,
        mut state: &mut State<RandomCardProvider>,
        is_starting_action_hit: bool,
    ) {
        if is_starting_action_hit {
            state.hit();

            while state.get_player_count() < 21 {
                let state_id = get_state_id(
                    &state.get_player_count(),
                    &state.get_dealer_showing(),
                    &state.get_usable_ace(),
                );
                let is_policy_hit = match policy.get(state_id.as_str()) {
                    Some(hit) => *hit,
                    None => false,
                };

                if !is_policy_hit {
                    break;
                }

                state.hit();
            }
        }
    }

    #[test]
    fn test_monte_carlo_exploring_starts_for_blackjack() {
        let iteration_count = 100000000;
        let discount_rate = 1.0;
        let card_provider: RandomCardProvider = RandomCardProvider::new();

        let player_count_range = 12_u8..=21;
        let dealer_showing_range = 2_u8..=11;
        let player_usable_aces_range = vec![true, false];

        let mut policy: HashMap<String, bool> = HashMap::new();
        let mut values: HashMap<String, (i32, f64)> = HashMap::new();

        (0..iteration_count).for_each(|_| {
            let starting_player_count = rand::rng().random_range(player_count_range.clone());
            let starting_dealer_showing = rand::rng().random_range(dealer_showing_range.clone());
            let starting_player_usable_aces = rand::rng().random_bool(0.5);

            let mut state = State::new(
                starting_player_count,
                starting_dealer_showing,
                starting_player_usable_aces,
                &card_provider,
            );
            let starting_state_id = get_state_id(
                &starting_player_count,
                &starting_dealer_showing,
                &starting_player_usable_aces,
            );
            let is_starting_action_hit = rand::rng().random_bool(0.5);

            play_episode(&policy, &mut state, is_starting_action_hit);
            let reward = state.check_for_win();

            let mut g = 0.0;

            state
                .get_previous_counts()
                .iter()
                .rev()
                .enumerate()
                .for_each(|(t, (player_count, usable_ace))| {
                    g = match t {
                        0 => reward,
                        _ => g,
                    };

                    let did_hit = match t {
                        0 => false,
                        _ => true,
                    };

                    if *player_count <= 21 {
                        let state_id =
                            get_state_id(player_count, &starting_dealer_showing, usable_ace);
                        let state_action_id = get_state_action_id(state_id.as_str(), did_hit);
                        let new_value = match values.get(&state_action_id) {
                            Some((count, current_average)) => (
                                count + 1,
                                crate::service::calc_average(*current_average, count + 1, g),
                            ),
                            None => (1, g),
                        };
                        values.insert(state_action_id, new_value);

                        let hit_id = get_state_action_id(state_id.as_str(), true);
                        let stay_id = get_state_action_id(state_id.as_str(), false);

                        let hit_value = match values.get(&hit_id) {
                            Some((count, value)) => value.clone(),
                            None => 0_f64,
                        };
                        let stay_value = match values.get(&stay_id) {
                            Some((count, value)) => value.clone(),
                            None => 0_f64,
                        };

                        if hit_value > stay_value {
                            policy.insert(state_id, true);
                        } else {
                            policy.insert(state_id, false);
                        }
                    }
                })
        });

        println!("usable ace");
        (11..=21).rev().for_each(|player_count| {
            let mut str = format!("player sum: {} | ", player_count);
            (2..=11).for_each(|dealer_showing| {
                let state_id = get_state_id(&player_count, &dealer_showing, &true);
                let policy_hit = match policy.get(state_id.as_str()) {
                    Some(hit) => *hit,
                    None => true,
                };
                let char = match policy_hit {
                    true => 'H',
                    false => 'S',
                };
                str.push_str(&format!("{} ", char));
            });
            println!("{}", str);
        });
        println!("no usable ace");
        (11..=21).rev().for_each(|player_count| {
            let mut str = format!("player sum: {} | ", player_count);
            (2..=11).for_each(|dealer_showing| {
                let state_id = get_state_id(&player_count, &dealer_showing, &false);
                let policy_hit = match policy.get(state_id.as_str()) {
                    Some(hit) => hit.clone(),
                    None => true,
                };
                let char = match policy_hit {
                    true => 'H',
                    false => 'S',
                };
                str.push_str(&format!("{} ", char));
            });
            println!("{}", str);
        });

        println!("dealer showing:  2 3 4 5 6 7 8 9 10A");
        println!("iterations: {}", iteration_count);
    }

    #[test]
    fn rev_a_list() {
        let mut list = vec![1, 2, 3, 4, 5];
        list.iter().enumerate().rev().for_each(|(i, v)| {
            println!("{}: {}", i, v);
        });
    }
}
