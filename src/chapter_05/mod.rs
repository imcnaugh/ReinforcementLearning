mod blackjack;
mod cards;

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path::PathBuf;
    use rand::Rng;
    use crate::chapter_05::blackjack::State;
    use crate::chapter_05::cards::RandomCardProvider;
    use crate::service::MultiLineChartBuilder;
    use crate::service::MultiLineChartData;

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
                    let mut state = State::new(player_start_count, dealer_showing_start, player_usable_aces, &card_provider);
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

        multi_line_cart_builder.set_path(PathBuf::from("output/chapter5/blackJack_values_static_policy.png")).set_title(format!("Blackjack Value"));
        multi_line_cart_builder.create_chart().unwrap();
    }

    fn get_state_action_id(state_id: &str, action: bool) -> String {
        format!("{}_{}", state_id, action)
    }

    fn play_episode(policy: &HashMap<String, bool>, mut state: &mut State<RandomCardProvider>, is_starting_action_hit: bool) {
        if is_starting_action_hit {
            state.hit();
            let is_policy_hit = match policy.get(&state.id()) {
                Some(hit) => *hit,
                None => false,
            };
            while state.get_player_count() < 21 && is_policy_hit {
                state.hit();
            }
        }
    }

    #[test]
    fn test_monte_carlo_exploring_starts_for_blackjack() {
        let iteration_count = 5;
        let discount_rate = 1.0;
        let card_provider: RandomCardProvider = RandomCardProvider::new();
        
        let player_count_range = 12_u8..=21;
        let dealer_showing_range = 2_u8..=11;
        let player_usable_aces_range = vec![true, false];

        let policy: HashMap<String, bool> = HashMap::new();
        let values: HashMap<String, f64> = HashMap::new();

        (0..iteration_count).for_each(|_| {
            let starting_player_count = rand::rng().random_range(player_count_range.clone());
            let starting_dealer_showing = rand::rng().random_range(dealer_showing_range.clone());
            let starting_player_usable_aces = rand::rng().random_bool(0.5);

            let mut state = State::new(starting_player_count, starting_dealer_showing, starting_player_usable_aces, &card_provider);
            let starting_state_id = state.id().clone();
            let is_starting_action_hit = rand::rng().random_bool(0.5);

            play_episode(&policy, &mut state, is_starting_action_hit);

            let mut g = 0.0;

        });
    }
}