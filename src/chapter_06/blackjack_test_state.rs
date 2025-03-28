use crate::chapter_05::cards::{CardProvider, RandomCardProvider, Value};
use crate::chapter_05::race_track::state::State;

pub struct BlackJackTestState {
    player_count: u8,
    dealer_showing: u8,
    usable_ace: bool,
    is_terminal: bool,
    state_value: f64,
}

impl BlackJackTestState {
    pub fn new(player_count: u8, dealer_showing: u8, usable_ace: bool, is_terminal: bool) -> Self {
        let state_value = if is_terminal { 0.0 } else { 0.0 }; // some arbitrary value, distinct from reality
        Self {
            player_count,
            dealer_showing,
            usable_ace,
            is_terminal,
            state_value,
        }
    }

    pub fn get_state_value(&self) -> f64 {
        self.state_value
    }

    pub fn set_state_value(&mut self, value: f64) {
        self.state_value = value;
    }
}

impl State for BlackJackTestState {
    fn get_id(&self) -> String {
        format!(
            "{}_{}_{}",
            self.player_count, self.usable_ace, self.dealer_showing
        )
    }

    fn get_actions(&self) -> Vec<String> {
        match self.is_terminal {
            true => vec!["hit".to_string(), "stand".to_string()],
            false => vec![],
        }
    }

    fn is_terminal(&self) -> bool {
        self.is_terminal
    }

    fn take_action(&self, action: &str) -> (f64, Self) {
        let card_provider = RandomCardProvider::new();
        match action {
            "hit" => {
                let new_card = card_provider.get_random_card().unwrap();

                let mut usable_ace = self.usable_ace || new_card == Value::Ace;
                let mut new_player_count = self.player_count + new_card.get_value();

                if new_player_count > 21 && usable_ace {
                    new_player_count = new_player_count - 10;
                    usable_ace = false;
                }

                let is_terminal = new_player_count > 21;

                let next_state = BlackJackTestState::new(
                    new_player_count,
                    self.dealer_showing,
                    usable_ace,
                    is_terminal,
                );
                let reward = if is_terminal { -1.0 } else { 0.0 };
                (reward, next_state)
            }
            "stand" => {
                let mut dealer_count = self.dealer_showing;
                let mut dealer_has_usable_ace = self.dealer_showing == 11;

                while dealer_count < 17 {
                    let new_card = card_provider.get_random_card().unwrap();
                    dealer_count = dealer_count + new_card.get_value();

                    if dealer_count > 21 && dealer_has_usable_ace {
                        dealer_count = dealer_count - 10;
                        dealer_has_usable_ace = false;
                    }

                    if dealer_count >= 17 {
                        break;
                    }
                }

                let reward = if dealer_count > 21 {
                    1.0
                } else if dealer_count == self.player_count {
                    0.0
                } else if dealer_count < self.player_count {
                    1.0
                } else {
                    -1.0
                };
                let next_state = BlackJackTestState::new(
                    self.player_count,
                    self.dealer_showing,
                    self.usable_ace,
                    true,
                );

                (reward, next_state)
            }
            _ => panic!("invalid action"),
        }
    }
}
