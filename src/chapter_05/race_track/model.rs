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
