use crate::chapter_05::race_track::track::TrackElement::{Finish, OutOfBounds, Start, Track};
use crate::chapter_05::race_track::track::{RaceTrack, TrackElement};
use crate::chapter_05::race_track::track_parser::TrackParseError::{
    FileError, UnexpectedCharacter,
};
use std::error::Error;
use std::fmt::{Debug, Display, Formatter};
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

#[derive(Debug)]
pub enum TrackParseError {
    FileError(io::Error),
    UnexpectedCharacter(char),
}

impl Display for TrackParseError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            FileError(e) => write!(f, "File error: {}", e),
            UnexpectedCharacter(c) => write!(f, "Unexpected character: {}", c),
        }
    }
}

impl Error for TrackParseError {}

pub fn parse_track_from_file(path: &Path) -> Result<RaceTrack, TrackParseError> {
    let path = Path::new(path);

    // Open the file in read-only mode.
    let file = File::open(&path).map_err(FileError)?;
    let reader = io::BufReader::new(file);

    let mut lines = Vec::new();
    for line in reader.lines() {
        let line = line.map_err(FileError)?;
        lines.push(line);
    }

    parse_track_from_lines(&lines)
}

pub fn parse_track_from_string(track_string: &str) -> Result<RaceTrack, TrackParseError> {
    let lines = track_string
        .lines()
        .map(|l| l.to_string())
        .collect::<Vec<String>>();
    parse_track_from_lines(&lines)
}

fn parse_track_from_lines(lines: &Vec<String>) -> Result<RaceTrack, TrackParseError> {
    let track = lines
        .iter()
        .rev()
        .map(|l| parse_line(l))
        .collect::<Result<Vec<Vec<TrackElement>>, TrackParseError>>()?;
    Ok(RaceTrack::new(track))
}

fn parse_line(line: &str) -> Result<Vec<TrackElement>, TrackParseError> {
    let mut track_elements: Vec<TrackElement> = Vec::new();
    for c in line.chars() {
        match c {
            ' ' => track_elements.push(Track),
            'X' => track_elements.push(OutOfBounds),
            'S' => track_elements.push(Start),
            'F' => track_elements.push(Finish),
            _ => return Err(UnexpectedCharacter(c)),
        }
    }
    Ok(track_elements)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_string_into_track_elements() {
        let input_str = "XS FX";
        let expected_track_elements = vec![OutOfBounds, Start, Track, Finish, OutOfBounds];

        let output = parse_line(input_str).unwrap();
        assert_eq!(
            output, expected_track_elements,
            "Failed to parse string into track elements"
        );
    }

    #[test]
    fn parse_invalid_string_into_track_elements() {
        let input_str = "Nope";

        let output = parse_line(input_str);

        match output {
            Ok(_) => panic!("Expected error"),
            Err(e) => match e {
                UnexpectedCharacter(c) => assert_eq!(c, 'N'),
                _ => panic!("Expected error of type UnexpectedCharacter, got: {}", e),
            },
        }
    }
}
