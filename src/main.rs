use std::io::{self, Write};
use ReinforcementLearning::chapter_05::blackjack::State;
use ReinforcementLearning::chapter_05::cards::{CardProvider, RandomCardProvider, Value::Ace};

fn main() {
    let card_provider = RandomCardProvider::new();

    let mut cards = vec![];
    let mut has_usable_ace = false;
    (0..2).for_each(|_| {
        let card = card_provider.get_random_card().unwrap();
        match card {
            Ace => has_usable_ace = true,
            _ => (),
        };
        cards.push(card);
    });

    let dealer_card = card_provider.get_random_card().unwrap();
    let value = cards.iter().fold(0, |acc, card| acc + card.get_value());

    let mut state = State::new(
        value,
        dealer_card.get_value(),
        has_usable_ace,
        &card_provider,
    );

    loop {
        if state.get_player_count() > 21 {
            println!("player went over 21");
            break;
        }

        println!("dealer is showing: {}", dealer_card.get_value());
        println!("player has: {}", state.get_player_count());
        cards.iter().for_each(|card| print!("{:?}, ", card));
        println!();

        print!("Enter 'h' to hit or 's' to stay:");
        io::stdout().flush().unwrap(); // Ensure the prompt is displayed before input
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let user_char = input.trim().chars().next().unwrap_or_default(); // Get the first character or default to '\0'

        match user_char {
            'h' => {
                let new_card = state.hit();
                println!("player was dealt: {:?}", new_card);
                cards.push(new_card);
            }
            's' => break,
            _ => (),
        }
        println!();
        println!();
    }

    let result = state.check_for_win();

    println!("{:?}", result);
}
