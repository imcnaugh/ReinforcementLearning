use rand;

pub struct Bandit {
    id: u32,
    reward: f32,
}

impl Bandit {
    pub fn new(id: u32, reward: f32) -> Bandit {
        Bandit { id, reward }
    }

    pub fn get_reward(&self) -> f32 {
        self.reward
    }
}
