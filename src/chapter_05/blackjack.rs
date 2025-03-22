use crate::chapter_05::cards::Value::Ace;
use crate::chapter_05::cards::{CardProvider, Value};
use std::fmt::{Display, Formatter};

pub struct BlackJackState<'a, P: CardProvider> {
    player_count: u8,
    dealer_showing: u8,
    usable_ace: bool,
    previous_counts: Vec<(u8, bool)>,
    card_provider: &'a P,
}

impl<'a, P: CardProvider> BlackJackState<'a, P> {
    pub fn new(
        player_count: u8,
        dealer_showing: u8,
        usable_ace: bool,
        card_provider: &'a P,
    ) -> Self {
        BlackJackState {
            player_count,
            dealer_showing,
            usable_ace,
            previous_counts: vec![(player_count, usable_ace)],
            card_provider,
        }
    }

    pub fn get_player_count(&self) -> u8 {
        self.player_count
    }

    pub fn get_usable_ace(&self) -> bool {
        self.usable_ace
    }

    pub fn get_dealer_showing(&self) -> u8 {
        self.dealer_showing
    }

    pub fn get_previous_counts(&self) -> &Vec<(u8, bool)> {
        &self.previous_counts
    }

    pub fn hit(&mut self) -> Value {
        let new_card = self.card_provider.get_random_card().unwrap();
        self.player_count = self.player_count + new_card.get_value();
        if self.player_count > 21 && self.usable_ace {
            self.player_count = self.player_count - 10;
            self.usable_ace = match new_card {
                Ace => true,
                _ => false,
            };
        }

        self.previous_counts
            .push((self.player_count, self.usable_ace));

        new_card
    }

    pub fn check_for_win(&self) -> f64 {
        if self.player_count > 21 {
            return -1.0;
        }

        let mut dealer_count = self.dealer_showing;
        if self.player_count == 21 && self.previous_counts.len() == 1 {
            let new_card = self.card_provider.get_random_card().unwrap();
            let new_card_value = Self::get_new_card_for_dealer(dealer_count, new_card);
            dealer_count = dealer_count + new_card_value;

            return if dealer_count == 21 {
                0.0
            } else {
                1.0
            }
        }

        while dealer_count < 17 {
            let new_card = self.card_provider.get_random_card().unwrap();
            let new_value = Self::get_new_card_for_dealer(dealer_count, new_card);
            dealer_count = dealer_count + new_value;
            // println!("dealer was dealt: {:?} is showing: {}", new_card, dealer_count);
        }

        if dealer_count > 21 {
            return 1.0;
        }

        if dealer_count < self.player_count {
            1.0
        } else if dealer_count > self.player_count {
            -1.0
        } else {
            0.0
        }
    }

    fn get_new_card_for_dealer(dealer_count: u8, new_card: Value) -> u8 {
        let new_value = match new_card {
            Ace => {
                if dealer_count + new_card.get_value() > 21 {
                    new_card.get_value() - 10
                } else {
                    new_card.get_value()
                }
            }
            _ => {
                new_card.get_value()
            }
        };
        new_value
    }

    pub fn print_previous_counts(&self) {
        println!("{:?}", self.previous_counts);
    }
}

impl<'a, P: CardProvider> Display for BlackJackState<'a, P> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Player: {}, with Ace: {}, dealer: {},",
            self.player_count, self.usable_ace, self.dealer_showing
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chapter_05::cards::Value;
    use crate::chapter_05::cards::Value::{Eight, Five, King, Seven, Six};

    struct MockCardProvider {
        card_to_return: Option<Value>,
    }

    impl MockCardProvider {
        fn new(card: Option<Value>) -> Self {
            Self {
                card_to_return: card,
            }
        }

        fn set_card_to_return(&mut self, card: Option<Value>) {
            self.card_to_return = card;
        }
    }

    impl CardProvider for MockCardProvider {
        fn get_random_card(&self) -> Result<Value, ()> {
            match self.card_to_return {
                Some(card) => Ok(card),
                None => Err(()),
            }
        }
    }

    #[test]
    fn test_hit() {
        let mock_card_provider = MockCardProvider::new(Some(Ace));

        let mut state = BlackJackState::new(5, 0, false, &mock_card_provider);
        state.hit();
        assert_eq!(state.player_count, 16);
        assert_eq!(state.usable_ace, true);
        assert_eq!(state.dealer_showing, 0);

        let mut state = BlackJackState::new(15, 0, false, &mock_card_provider);
        state.hit();
        assert_eq!(state.player_count, 16);
        assert_eq!(state.usable_ace, false);
        assert_eq!(state.dealer_showing, 0);

        let mut state = BlackJackState::new(15, 0, true, &mock_card_provider);
        state.hit();
        assert_eq!(state.player_count, 16);
        assert_eq!(state.usable_ace, true);
        assert_eq!(state.dealer_showing, 0);

        let mock_card_provider = MockCardProvider::new(Some(King));
        let mut state = BlackJackState::new(15, 0, false, &mock_card_provider);
        state.hit();
        state.hit();
        assert_eq!(state.player_count, 35);
        assert_eq!(state.usable_ace, false);
        assert_eq!(state.dealer_showing, 0);
    }

    #[test]
    fn test_dealer_will_not_hit_above_16() {
        let mut mock_card_provider = MockCardProvider::new(Some(Seven));
        let state = BlackJackState::new(18, 10, false, &mock_card_provider);
        let result = state.check_for_win();
        assert_eq!(result, 1.0);

        mock_card_provider.set_card_to_return(Some(Eight));
        let state = BlackJackState::new(18, 10, true, &mock_card_provider);
        let result = state.check_for_win();
        assert_eq!(result, 0.0);
    }
}
