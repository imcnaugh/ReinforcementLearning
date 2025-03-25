pub struct Racer {
    position: (usize, usize),
    velocity: (i32, i32),
    max_velocity: i32,
}

impl Racer {
    pub fn new(starting_position: (usize, usize)) -> Self {
        Self {
            position: starting_position,
            velocity: (0, 0),
            max_velocity: 5,
        }
    }

    pub fn get_velocity(&self) -> (i32, i32) {
        self.velocity
    }

    pub fn get_position(&self) -> (usize, usize) {
        self.position
    }

    pub fn increase_horizontal_velocity(&mut self) {
        self.velocity.0 = self.max_velocity.min(self.velocity.0 + 1);
    }

    pub fn decrease_horizontal_velocity(&mut self) {
        self.velocity.0 = -self.max_velocity.max(self.velocity.0 - 1);
    }

    pub fn increase_vertical_velocity(&mut self) {
        self.velocity.1 = self.max_velocity.min(self.velocity.1 + 1);
    }

    pub fn decrease_vertical_velocity(&mut self) {
        self.velocity.1 = -self.max_velocity.max(self.velocity.1 - 1);
    }
}
