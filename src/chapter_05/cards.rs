use rand::Rng;

#[derive(Copy, Clone, Debug)]
pub enum Value {
    Ace,
    Two,
    Three,
    Four,
    Five,
    Six,
    Seven,
    Eight,
    Nine,
    Ten,
    Jack,
    Queen,
    King,
}

impl Value {
    pub fn get_value(&self) -> u8 {
        match self {
            Value::Two => 2,
            Value::Three => 3,
            Value::Four => 4,
            Value::Five => 5,
            Value::Six => 6,
            Value::Seven => 7,
            Value::Eight => 8,
            Value::Nine => 9,
            Value::Ten => 10,
            Value::Jack => 10,
            Value::Queen => 10,
            Value::King => 10,
            Value::Ace => 11,
        }
    }
}

pub trait CardProvider {
    fn get_random_card(&self) -> Result<Value, ()>;
}

pub struct RandomCardProvider;

impl RandomCardProvider {
    pub fn new() -> Self {
        RandomCardProvider
    }
}

impl CardProvider for RandomCardProvider {
    fn get_random_card(&self) -> Result<Value, ()> {
        let values = [
            Value::Ace,
            Value::Two,
            Value::Three,
            Value::Four,
            Value::Five,
            Value::Six,
            Value::Seven,
            Value::Eight,
            Value::Nine,
            Value::Ten,
            Value::Jack,
            Value::Queen,
            Value::King,
        ];

        let index = rand::rng().random_range(0..values.len());
        Ok(values[index])
    }
}
