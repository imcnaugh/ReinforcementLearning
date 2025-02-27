use crate::chapter_02::Bandit;

pub struct KArmedBandit {
    bandits: Vec<Bandit>,
}

impl KArmedBandit {
    pub fn new(bandits: Vec<Bandit>) -> Self {
        Self { bandits }
    }

    pub fn rand_new(count: u32) -> Self {
        let bandits = (0..count)
            .map(|i| Bandit::new(i, rand::random::<f32>()))
            .collect();
        Self { bandits }
    }

    pub fn get_bandit(&self) -> &Vec<Bandit> {
        &self.bandits
    }
}
