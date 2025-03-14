mod blackjack;
mod cards;

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
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

    fn calc_average(
        current_average: f64,
        total_count: i32,
        new_reward: f64,
    ) -> f64 {
        current_average + (1.0 / total_count as f64) * (new_reward - current_average)
    }

    #[test]
    fn plot_blackjack_state_value() {
        let mut multi_line_cart_builder = MultiLineChartBuilder::new();
        let card_provider: RandomCardProvider = RandomCardProvider::new();

        (2..=11).for_each(|dealer_showing| {
            let mut average_rewards: Vec<f64> = vec![];
            (12..=21).for_each(|player_count| {
                let player_start_count = player_count;
                let player_usable_aces = false;
                let dealer_showing_start = dealer_showing;

                let mut running_average: f64 = 0.00;

                (0..500000).for_each(|i| {
                    let mut state = State::new(player_start_count, dealer_showing_start, player_usable_aces, &card_provider);
                    hit_unless_above_20(&mut state);
                    let reward = state.check_for_win();
                    running_average = calc_average(running_average, i + 1, reward);
                });

                average_rewards.push(running_average);
            });
            let mut multi_line_chart_data = MultiLineChartData::new(average_rewards);
            multi_line_chart_data.set_label(format!("Dealer shows {}", dealer_showing));
            multi_line_cart_builder.add_data(multi_line_chart_data);
        });

        multi_line_cart_builder.set_path(PathBuf::from("output/chapter5/blackJack_values_static_policy.png"));
        multi_line_cart_builder.create_chart().unwrap();
    }
}