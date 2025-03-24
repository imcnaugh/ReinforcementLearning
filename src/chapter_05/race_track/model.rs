use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackElement {
    Track,
    OutOfBounds,
    Start,
    Finish,
}

pub struct RaceTrack {
    track: Vec<Vec<TrackElement>>,
    start_positions: Vec<(usize, usize)>,
    finish_positions: Vec<(usize, usize)>,
}

impl RaceTrack {
    pub fn new(track: Vec<Vec<TrackElement>>) -> Self {
        let mut start_positions = vec![];
        let mut finish_positions = vec![];

        track.iter().enumerate().for_each(|(y, row)| {
            row.iter()
                .enumerate()
                .for_each(|(x, element)| match element {
                    TrackElement::Start => start_positions.push((x, y)),
                    TrackElement::Finish => finish_positions.push((x, y)),
                    _ => (),
                })
        });

        Self {
            track,
            start_positions,
            finish_positions,
        }
    }

    fn check_for_intersections(
        &self,
        start_position: (usize, usize),
        vertical_velocity: i32,
        horizontal_velocity: i32,
    ) -> Option<TrackElement> {
        if horizontal_velocity == 0 {
            for i in 0..vertical_velocity.abs() as usize {
                let y = start_position.1 as i32 + (i as i32 * vertical_velocity.signum());
                let x = start_position.0 as i32;
                if y < 0 || y as usize >= self.track.len() {
                    return Some(TrackElement::OutOfBounds);
                }
                match self.track[y as usize][x as usize] {
                    TrackElement::OutOfBounds => {
                        return Some(TrackElement::OutOfBounds);
                    }
                    TrackElement::Finish => {
                        return Some(TrackElement::Finish);
                    }
                    _ => (),
                };
            }
            return None;
        }

        let slope = vertical_velocity as f32 / horizontal_velocity as f32;
        for i in 0..=horizontal_velocity.abs() as usize {
            let x = start_position.0 as i32 + (i as i32 * horizontal_velocity.signum());
            let y = (slope * (i as i32 * vertical_velocity.signum()) as f32) as i32 + start_position.1 as i32;
            if x < 0 || x as usize >= self.track[0].len() {
                return Some(TrackElement::OutOfBounds);
            }

            if y < 0 || y as usize >= self.track.len() {
                return Some(TrackElement::OutOfBounds);
            }
            match self.track[y as usize][x as usize] {
                TrackElement::OutOfBounds => {
                    return Some(TrackElement::OutOfBounds);
                },
                TrackElement::Finish => {
                    return Some(TrackElement::Finish);
                },
                _ => (),
            };
        }
        None
    }
}

impl Display for RaceTrack {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut str = String::new();
        self.track.iter().enumerate().for_each(|(y, row)| {
            row.iter().enumerate().for_each(|(x, element)| {
                let element_as_char = match element {
                    TrackElement::Track => ' ',
                    TrackElement::OutOfBounds => 'X',
                    TrackElement::Start => 'S',
                    TrackElement::Finish => 'F',
                };
                str.push(element_as_char);
            });
            str.push('\n');
        });
        write!(f, "{}", str)
    }
}

#[cfg(test)]
mod tests {
    use crate::chapter_05::race_track::track_parser::parse_track_from_string;
    use super::*;

    #[test]
    fn test_check_for_intersections() {
        let track_string = " XX\nX X\nXX ".to_string();
        let track = parse_track_from_string(&track_string).unwrap();
        let start_position = (0, 0);
        let vertical_velocity = 2;
        let horizontal_velocity = 2;

        println!("{}", track);

        let output = track.check_for_intersections(start_position, vertical_velocity, horizontal_velocity);

        match output {
            None => assert!(true),
            Some(track_element) => panic!("Expected no intersection but found {:?}", track_element),
        }
    }

    #[test]
    fn test_check_for_intersections_no_horizontal_velocity() {
        let track_string = "\
        X X\n\
        X X".to_string();
        let track = parse_track_from_string(&track_string).unwrap();
        let start_position = (1, 0);
        let vertical_velocity = 2;
        let horizontal_velocity = 0;

        let output = track.check_for_intersections(start_position, vertical_velocity, horizontal_velocity);

        println!("{}", track);

        match output {
            None => assert!(true),
            Some(element) => panic!("Expected no intersection but found {:?}", element),
        }
    }
}