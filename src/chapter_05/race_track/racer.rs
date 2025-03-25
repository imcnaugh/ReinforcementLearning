use crate::chapter_05::race_track::state::State;
use crate::chapter_05::race_track::track::{RaceTrack, TrackElement};

#[derive(Clone)]
pub struct Racer<'a> {
    position: (i32, i32),
    velocity: (i32, i32),
    max_velocity: i32,
    crossed_finish_line: bool,
    track: &'a RaceTrack,
}

impl<'a> Racer<'_> {
    pub fn new(starting_position: (i32, i32), track: &'a RaceTrack) -> Racer {
        Racer {
            position: starting_position,
            track,
            crossed_finish_line: false,
            velocity: (0, 0),
            max_velocity: 5,
        }
    }

    pub fn get_velocity(&self) -> (i32, i32) {
        self.velocity
    }

    pub fn get_position(&self) -> (i32, i32) {
        self.position
    }

    pub fn increase_horizontal_velocity(&mut self) {
        self.velocity.0 = self.max_velocity.min(self.velocity.0 + 1);
    }

    pub fn decrease_horizontal_velocity(&mut self) {
        self.velocity.0 = (self.max_velocity * -1).max(self.velocity.0 - 1);
    }

    pub fn increase_vertical_velocity(&mut self) {
        self.velocity.1 = self.max_velocity.min(self.velocity.1 + 1);
    }

    pub fn decrease_vertical_velocity(&mut self) {
        self.velocity.1 = (self.max_velocity * -1).max(self.velocity.1 - 1);
    }
}

impl State for Racer<'_> {
    fn get_id(&self) -> String {
        format!(
            "{}_{}_{}_{}",
            self.position.0, self.position.1, self.velocity.0, self.velocity.1
        )
    }

    fn get_actions(&self) -> Vec<String> {
        let can_increase_horizontal_velocity = self.velocity.0 < self.max_velocity;
        let can_decrease_horizontal_velocity = self.velocity.0 > -self.max_velocity;
        let can_increase_vertical_velocity = self.velocity.1 < self.max_velocity;
        let can_decrease_vertical_velocity = self.velocity.1 > -self.max_velocity;

        let mut response: Vec<String> = Vec::new();
        if can_increase_horizontal_velocity {
            let base = "h+";
            response.push(String::from(base));
            if can_increase_vertical_velocity {
                response.push(format!("{}_v+", base));
            }
            if can_decrease_vertical_velocity {
                response.push(format!("{}_v-", base));
            }
        }
        if can_decrease_horizontal_velocity {
            let base = "h-";
            response.push(String::from(base));
            if can_increase_vertical_velocity {
                response.push(format!("{}_v+", base));
            }
            if can_decrease_vertical_velocity {
                response.push(format!("{}_v-", base));
            }
        }
        if can_increase_vertical_velocity {
            response.push(String::from("v+"));
        }
        if can_decrease_vertical_velocity {
            response.push(String::from("v-"));
        }
        response
    }

    fn is_terminal(&self) -> bool {
        self.crossed_finish_line
    }

    fn take_action(&self, action: &str) -> (f64, Self) {
        let mut new_state = self.clone();
        if action.contains("h+") {
            new_state.increase_horizontal_velocity();
        }
        if action.contains("h-") {
            new_state.decrease_horizontal_velocity();
        }
        if action.contains("v+") {
            new_state.increase_vertical_velocity();
        }
        if action.contains("v-") {
            new_state.decrease_vertical_velocity();
        }

        let reward = match self.track.check_for_intersections(
            new_state.position,
            new_state.velocity.0,
            new_state.velocity.1,
        ) {
            None => -1.0_f64,
            Some(element) => match element {
                TrackElement::OutOfBounds => {
                    new_state.crossed_finish_line = true;
                    -1000.0_f64
                }
                TrackElement::Finish => {
                    new_state.crossed_finish_line = true;
                    -1.0_f64
                }
                _ => -1.0_f64,
            },
        };

        new_state.position.0 += new_state.velocity.0;
        new_state.position.1 += new_state.velocity.1;

        (reward, new_state)
    }
}
