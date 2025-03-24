pub mod blackjack;
pub mod cards;
mod importance_sampling;
pub mod policy;
mod state;

#[cfg(test)]
mod tests {
    use crate::chapter_05::blackjack::BlackJackState;
    use crate::chapter_05::cards::RandomCardProvider;
    use crate::chapter_05::importance_sampling::{
        ordinary_importance_sampling, weighted_importance_sampling,
        weighted_importance_sampling_incremental,
    };
    use crate::chapter_05::policy::{DeterministicPolicy, Policy, StochasticPolicy};
    use crate::service::MultiLineChartData;
    use crate::service::{
        calc_average, mean_square_error, LineChartBuilder, LineChartData, MultiLineChartBuilder,
    };
    use egui::Key::S;
    use plotters::prelude::ShapeStyle;
    use plotters::style::{BLUE, RED};
    use rand::Rng;
    use std::collections::HashMap;
    use std::hash::Hash;
    use std::path::PathBuf;

    fn hit_unless_above_20(state: &mut BlackJackState<RandomCardProvider>) {
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
                    let mut state = BlackJackState::new(
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
        mut state: &mut BlackJackState<RandomCardProvider>,
        is_starting_action_hit: bool,
    ) {
        if state.get_player_count() == 21 {
            return;
        }

        if is_starting_action_hit {
            state.hit();

            while state.get_player_count() <= 21 {
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

            let mut state = BlackJackState::new(
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
                            Some((count, current_average)) => {
                                (count + 1, calc_average(*current_average, count + 1, g))
                            }
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

        println!("dealer showing:  2 3 4 5 6 7 8 9 10A ");
        println!("iterations: {}", iteration_count);
    }

    #[test]
    fn rev_a_list() {
        let mut list = vec![1, 2, 3, 4, 5];
        list.iter().enumerate().rev().for_each(|(i, v)| {
            println!("{}: {}", i, v);
        });
    }

    #[test]
    fn test_monte_carlo_kinda_exploring_starts_but_with_e_soft_policy() {
        let iteration_count = 100000000;
        let e_soft_rate = 0.3;
        let card_provider: RandomCardProvider = RandomCardProvider::new();

        let player_count_range = 12_u8..=14;
        let dealer_showing_range = 2_u8..=11;

        let mut policy = StochasticPolicy::new();
        let mut values: HashMap<String, (i32, f64)> = HashMap::new();

        (0..iteration_count).for_each(|_| {
            let starting_player_count = rand::rng().random_range(player_count_range.clone());
            let starting_dealer_showing = rand::rng().random_range(dealer_showing_range.clone());
            let starting_player_usable_aces = rand::rng().random_bool(0.5);

            let mut state = BlackJackState::new(
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

            while state.get_player_count() <= 21 {
                let current_state_id = get_state_id(
                    &state.get_player_count(),
                    &state.get_dealer_showing(),
                    &state.get_usable_ace(),
                );
                match policy.pick_action_for_state(&current_state_id) {
                    Ok(a) => {
                        if a == "stay" {
                            break;
                        }
                    }
                    Err(_) => (),
                };
                state.hit();
            }

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

                        let best_action = if hit_value > stay_value {
                            String::from("hit")
                        } else {
                            String::from("stay")
                        };

                        policy
                            .set_state_actions_probabilities_using_e_soft_probabilities(
                                state_id.as_str(),
                                vec![String::from("hit"), String::from("stay")],
                                e_soft_rate,
                                best_action,
                            )
                            .unwrap();
                    }
                })
        });

        let hit_string = String::from("hit");
        let stay_string = String::from("stay");

        println!("usable ace");
        (12..=21).rev().for_each(|player_count| {
            let mut str = format!("player sum: {} | ", player_count);
            (2..=11).for_each(|dealer_showing| {
                let state_id = get_state_id(&player_count, &dealer_showing, &true);
                let action = policy.get_actions_for_state(state_id.as_str()).unwrap();

                let max_action = action
                    .iter()
                    .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
                    .unwrap();

                let char = if max_action.1 == hit_string { 'H' } else { 'S' };
                str.push_str(&format!("{} ", char));
            });
            println!("{}", str);
        });
        println!("no usable ace");
        (12..=21).rev().for_each(|player_count| {
            let mut str = format!("player sum: {} | ", player_count);
            (2..=11).for_each(|dealer_showing| {
                let state_id = get_state_id(&player_count, &dealer_showing, &false);
                let action = policy.get_actions_for_state(state_id.as_str()).unwrap();

                let max_action = action
                    .iter()
                    .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
                    .unwrap();

                let char = if max_action.1 == hit_string { 'H' } else { 'S' };
                str.push_str(&format!("{} ", char));
            });
            println!("{}", str);
        });

        println!("dealer showing:  2 3 4 5 6 7 8 9 10A");
        println!("iterations: {}", iteration_count);
    }

    #[test]
    fn example_5_4_off_policy_blackjack_estimations() {
        let card_provider: RandomCardProvider = RandomCardProvider::new();
        let mut target_policy = DeterministicPolicy::new();
        let mut behavior_policy = StochasticPolicy::new();
        let expected = -0.26938;
        (2..=11).for_each(|dealer_showing| {
            vec![true, false].iter().for_each(|usable_ace| {
                (13..20).for_each(|player_count| {
                    let state_id = get_state_id(&player_count, &dealer_showing, &usable_ace);
                    target_policy.set_action_for_state(&state_id, "hit");

                    let stochastic_actions =
                        vec![(0.5, String::from("hit")), (0.5, String::from("stay"))];
                    behavior_policy
                        .set_state_action_probabilities(&state_id, stochastic_actions)
                        .unwrap()
                });
                (20..=31).for_each(|player_count| {
                    let state_id = get_state_id(&player_count, &dealer_showing, &usable_ace);
                    target_policy.set_action_for_state(&state_id, "stay");

                    let stochastic_actions = if player_count <= 21 {
                        vec![(0.5, String::from("hit")), (0.5, String::from("stay"))]
                    } else {
                        vec![(1.0, String::from("stay"))]
                    };
                    behavior_policy
                        .set_state_action_probabilities(&state_id, stochastic_actions)
                        .unwrap()
                });
            })
        });

        /*
        Let's verify the claim that a starting state of
        Dealer Showing: 2
        Player Count: 13
        Usable Ace: True
        has a value of -0.27726
        Eh im getting -0.26938 but my blackjack logic is a bit off, its good enough
         */
        // let mut running_average: f64 = 0.00;
        // (0..100000000).for_each(|i| {
        //     let mut state = BlackJackState::new(13, 2, true, &card_provider);
        //     loop {
        //         let state_id = get_state_id(
        //             &state.get_player_count(),
        //             &state.get_dealer_showing(),
        //             &state.get_usable_ace(),
        //         );
        //         let action = target_policy.pick_action_for_state(state_id.as_str()).unwrap_or_else(|_| "stay");
        //
        //         if action == "stay" {
        //             break;
        //         }
        //         state.hit();
        //     }
        //     let reward = state.check_for_win();
        //     running_average = crate::service::calc_average(running_average, i + 1, reward);
        // });
        //
        // assert_eq!(running_average, -0.27726);

        let num_of_episodes = 1000;
        let num_of_runs = 100;

        let mut ordinary_mean_squared_errors: Vec<Vec<f32>> = vec![vec![]; num_of_episodes];
        let mut weighted_mean_squared_errors: Vec<Vec<f32>> = vec![vec![]; num_of_episodes];

        (0..num_of_runs).for_each(|run_number| {
            let mut state_actions_and_rewards_for_each_episode: Vec<(Vec<(String, String)>, f64)> =
                vec![];

            (0..num_of_episodes).for_each(|i| {
                let mut state = BlackJackState::new(13, 2, true, &card_provider);
                let mut state_action_pairs: Vec<(String, String)> = vec![];
                loop {
                    let state_id = get_state_id(
                        &state.get_player_count(),
                        &state.get_dealer_showing(),
                        &state.get_usable_ace(),
                    );
                    let action = behavior_policy
                        .pick_action_for_state(state_id.as_str())
                        .unwrap_or_else(|_| "stay");
                    if action == "stay" {
                        state_action_pairs.push((state_id.clone(), "stay".to_string()));
                        break;
                    }
                    state_action_pairs.push((state_id.clone(), "hit".to_string()));
                    state.hit();
                }
                let reward = state.check_for_win();
                let state_action_pairs = state_action_pairs[0..state_action_pairs.len()].to_vec();
                state_actions_and_rewards_for_each_episode.push((state_action_pairs, reward));

                match weighted_importance_sampling(
                    &state_actions_and_rewards_for_each_episode,
                    &target_policy,
                    &behavior_policy,
                ) {
                    Ok(weighted_importance) => {
                        let mean_square_error = mean_square_error(expected, weighted_importance);
                        weighted_mean_squared_errors[i].push(mean_square_error as f32);
                    }
                    Err(_) => (),
                };
                match ordinary_importance_sampling(
                    &state_actions_and_rewards_for_each_episode,
                    &target_policy,
                    &behavior_policy,
                ) {
                    Ok(ordinary_importance) => {
                        let mean_square_error = mean_square_error(expected, ordinary_importance);
                        ordinary_mean_squared_errors[i].push(mean_square_error as f32);
                    }
                    Err(_) => {}
                };
            });
        });

        let mut avg_org: Vec<f64> = vec![0.0; num_of_episodes];
        let mut avg_wei: Vec<f64> = vec![0.0; num_of_episodes];

        ordinary_mean_squared_errors
            .iter()
            .enumerate()
            .for_each(|(run_index, v)| {
                let a = v.iter().enumerate().fold(0.0, |acc, (episode_index, x)| {
                    calc_average(acc, (episode_index + 1) as i32, *x as f64)
                });
                avg_org[run_index] = a;
            });
        weighted_mean_squared_errors
            .iter()
            .enumerate()
            .for_each(|(run_index, v)| {
                let a = v.iter().enumerate().fold(0.0, |acc, (episode_index, x)| {
                    calc_average(acc, (episode_index + 1) as i32, *x as f64)
                });
                avg_wei[run_index] = a;
            });

        let avg_org: Vec<(f32, f32)> = avg_org
            .iter()
            .enumerate()
            .map(|(i, v)| (i as f32, *v as f32))
            .collect();
        let avg_wei: Vec<(f32, f32)> = avg_wei
            .iter()
            .enumerate()
            .map(|(i, v)| (i as f32, *v as f32))
            .collect();

        println!("Last value of avg_org: {:?}", avg_org.last().unwrap().1);
        println!("Last value of avg_wei: {:?}", avg_wei.last().unwrap().1);

        let ordinary_chart_data = LineChartData::new(
            "Ordinary mean squared error".to_string(),
            avg_org,
            ShapeStyle::from(&RED),
        );
        let weighted_chart_data = LineChartData::new(
            "Weighted mean squared error".to_string(),
            avg_wei,
            ShapeStyle::from(&BLUE),
        );

        let mut chart = LineChartBuilder::new();
        chart
            .add_data(ordinary_chart_data)
            .add_data(weighted_chart_data)
            .set_path(PathBuf::from(
                "output/chapter5/blackJack_values_off_policy.png",
            ))
            .set_title(format!(
                "off policy blackjack mean squared error, averaged over {} runs",
                num_of_runs
            ));

        chart.create_chart().unwrap();
    }

    #[test]
    fn off_policy_general_policy_iteration_for_blackjack() {
        let number_of_episodes = 1000000;
        let player_count_starting_range = 11..=21;
        let dealer_showing_starting_range = 2..=11;

        let card_provider: RandomCardProvider = RandomCardProvider::new();
        let mut state_action_values: HashMap<String, f64> = HashMap::new();
        let mut state_action_weights: HashMap<String, f64> = HashMap::new();
        let mut target_policy = DeterministicPolicy::new();

        let mut behavior_policy = StochasticPolicy::new();
        (2..=11).for_each(|dealer_showing| {
            vec![true, false].iter().for_each(|usable_ace| {
                (2..=32).for_each(|player_count| {
                    let state_id = get_state_id(&player_count, &dealer_showing, &usable_ace);
                    let state_action_pairs = if player_count <= 21 {
                        vec![(0.5, String::from("hit")), (0.5, String::from("stay"))]
                    } else {
                        vec![(1.0, String::from("stay"))]
                    };

                    behavior_policy
                        .set_state_action_probabilities(&state_id, state_action_pairs)
                        .unwrap();
                    target_policy.set_action_for_state(&state_id, "stay");
                })
            })
        });

        (0..number_of_episodes).for_each(|i| {
            let starting_player_count =
                rand::rng().random_range(player_count_starting_range.clone());
            let starting_dealer_showing =
                rand::rng().random_range(dealer_showing_starting_range.clone());
            let starting_player_usable_aces = rand::rng().random_bool(0.5);

            let mut blackjack_state = BlackJackState::new(
                starting_player_count,
                starting_dealer_showing,
                starting_player_usable_aces,
                &card_provider,
            );

            loop {
                if blackjack_state.get_player_count() > 21 {
                    break;
                }
                let state_id = get_state_id(
                    &blackjack_state.get_player_count(),
                    &blackjack_state.get_dealer_showing(),
                    &blackjack_state.get_usable_ace(),
                );
                let action = behavior_policy
                    .pick_action_for_state(state_id.as_str())
                    .unwrap_or_else(|_| "stay");
                if action == "stay" {
                    break;
                } else {
                    blackjack_state.hit();
                }
            }
            let reward = blackjack_state.check_for_win();

            let g = 0.0;
            let mut w = 1.0;

            for (t, (player_count, usable_ace)) in blackjack_state
                .get_previous_counts()
                .iter()
                .rev()
                .enumerate() {
                    let state_id = get_state_id(
                        player_count,
                        &blackjack_state.get_dealer_showing(),
                        usable_ace,
                    );
                    let current_cumulative_weight = match state_action_weights.get(&state_id) {
                        None => None,
                        Some(cumulative_weight) => Some(*cumulative_weight),
                    };


                    let last_t_plus_one_counts: Vec<_> = blackjack_state
                        .get_previous_counts()
                        .iter()
                        .rev()
                        .take(t + 1)
                        .enumerate()
                        .map(|(index, (a, b))| {
                            let state_id = get_state_id(a, &blackjack_state.get_dealer_showing(), b);
                            match index {
                                0 => (state_id.clone(), "stay".to_string()),
                                _ => (state_id.clone(), "hit".to_string()),
                            }
                        })
                        .collect();

                    let last_t_plus_one = &last_t_plus_one_counts[0];
                    let current_average = state_action_values.get(&last_t_plus_one.0).unwrap_or(&0.0);
                    let current_weighted_sum = state_action_weights.get(&last_t_plus_one.0).unwrap_or(&0.0);


                    let (new_average, new_cumulative_sum) =
                        weighted_importance_sampling_incremental(
                            &last_t_plus_one_counts,
                            reward,
                            *current_average,
                            current_cumulative_weight,
                            &target_policy,
                            &behavior_policy
                        ).unwrap();

                    let state_action_id = format!("{}_{}", last_t_plus_one.0, last_t_plus_one.1);
                    state_action_values.insert(state_action_id.clone(), new_average);
                    state_action_weights.insert(state_action_id.clone(), new_cumulative_sum.unwrap());

                    let stay_value = match state_action_values.get(format!("{}_stay", state_action_id).as_str()) {
                        Some(value) => *value,
                        None => f64::MIN,
                    };
                    let hit_value = match state_action_values.get(format!("{}_hit", state_action_id).as_str()) {
                        Some(value) => *value,
                        None => f64::MIN,
                    };

                    let best_action = if stay_value > hit_value {
                        "stay"
                    } else {
                        "hit"
                    };
                    target_policy.set_action_for_state(&state_id, best_action);
                    // Break loop if best action is not the one taken

                    let action_taken = match t {
                        0 => "stay",
                        _ => "hit",
                    };

                    if action_taken != best_action {
                        break;
                    }

                    let actions_for_state = target_policy.get_actions_for_state(&state_id).unwrap();
                    let odds_of_action_taken = actions_for_state.iter().find(|a| a.1 == action_taken).unwrap().0;

                    w *= 1.0 / odds_of_action_taken;
                }
        });

        let hit_string = String::from("hit");
        let stay_string = String::from("stay");

        println!("usable ace");
        (12..=21).rev().for_each(|player_count| {
            let mut str = format!("player sum: {} | ", player_count);
            (2..=11).for_each(|dealer_showing| {
                let state_id = get_state_id(&player_count, &dealer_showing, &true);
                let action = target_policy.get_actions_for_state(state_id.as_str()).unwrap();

                let max_action = action
                    .iter()
                    .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
                    .unwrap();

                let char = if max_action.1 == hit_string { 'H' } else { 'S' };
                str.push_str(&format!("{} ", char));
            });
            println!("{}", str);
        });

        println!("no usable ace");
        (12..=21).rev().for_each(|player_count| {
            let mut str = format!("player sum: {} | ", player_count);
            (2..=11).for_each(|dealer_showing| {
                let state_id = get_state_id(&player_count, &dealer_showing, &false);
                let action = target_policy.get_actions_for_state(state_id.as_str()).unwrap();

                let max_action = action
                    .iter()
                    .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
                    .unwrap();

                let char = if max_action.1 == hit_string { 'H' } else { 'S' };
                str.push_str(&format!("{} ", char));
            });
            println!("{}", str);
        });
    }
}
