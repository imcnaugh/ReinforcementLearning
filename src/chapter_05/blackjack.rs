use std::fmt::{Display, Formatter};
use crate::chapter_05::cards::CardProvider;
use crate::chapter_05::cards::Value::Ace;

pub struct State<'a, P: CardProvider> {
    player_count: u8,
    dealer_showing: u8,
    usable_ace: bool,
    card_provider: &'a P,
}

impl<'a, P: CardProvider> State<'a, P> {
    pub fn new(player_count: u8, dealer_showing: u8, usable_ace: bool, card_provider: &'a P) -> Self {
        State {
            player_count,
            dealer_showing,
            usable_ace,
            card_provider,
        }
    }

    pub fn hit(&mut self) {
        let new_card = self.card_provider.get_random_card().unwrap();
        let new_player_count = self.player_count + new_card.get_value();
        self.player_count = new_player_count;
        
        if new_player_count > 21 {
            if self.usable_ace {
                self.player_count = self.player_count - 10;
            } else {
                match new_card {
                    Ace => {
                        self.player_count = self.player_count - 10;
                        self.usable_ace = false;
                    },
                    _ => (),
                }
            }
        } else {
            match new_card {
                Ace => {
                    self.usable_ace = true;
                },
                _ => (),
            }
        }
    }
}

impl<'a, P: CardProvider> Display for State<'a, P> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Player: {}, with Ace: {}, dealer: {},", self.player_count, self.usable_ace, self.dealer_showing)
    }
}

#[cfg(test)]
mod tests {
    use crate::chapter_05::cards::Value;
    use super::*;

    struct MockCardProvider{
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

        let mut state = State::new(5, 0, false, &mock_card_provider);
        state.hit();
        assert_eq!(state.player_count, 16);
        assert_eq!(state.usable_ace, true);
        assert_eq!(state.dealer_showing, 0);

        let mut state = State::new(15, 0, false, &mock_card_provider);
        state.hit();
        assert_eq!(state.player_count, 16);
        assert_eq!(state.usable_ace, false);
        assert_eq!(state.dealer_showing, 0);

        let mut state = State::new(15, 0, true, &mock_card_provider);
        state.hit();
        assert_eq!(state.player_count, 16);
        assert_eq!(state.usable_ace, true);
        assert_eq!(state.dealer_showing, 0);
    }
}