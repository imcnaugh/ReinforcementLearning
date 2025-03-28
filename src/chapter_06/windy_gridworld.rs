use crate::attempts_at_framework::v1::state::State;
use std::collections::HashMap;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, Copy)]
pub enum Direction {
    North,
    East,
    South,
    West,
}

impl Display for Direction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Direction::North => write!(f, "North"),
            Direction::East => write!(f, "East"),
            Direction::South => write!(f, "South"),
            Direction::West => write!(f, "West"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct WindyGridworld {
    height: i16,
    width: i16,
}

impl WindyGridworld {
    pub fn new(height: i16, width: i16) -> Self {
        Self { height, width }
    }

    pub fn make_state_for_row_col(&self, row: u8, col: u8) -> WindyGridworldState {
        let mut wind_map_by_col: HashMap<u8, (u8, Direction)> = HashMap::new();
        wind_map_by_col.insert(3, (1, Direction::North));
        wind_map_by_col.insert(4, (1, Direction::North));
        wind_map_by_col.insert(5, (1, Direction::North));
        wind_map_by_col.insert(6, (2, Direction::North));
        wind_map_by_col.insert(7, (2, Direction::North));
        wind_map_by_col.insert(8, (1, Direction::North));

        let wind = wind_map_by_col.get(&col);

        let is_terminal = row == 3 && col == 7;

        WindyGridworldState::new(row, col, self, is_terminal, wind.cloned())
    }
}

#[derive(Debug, Clone)]
pub struct WindyGridworldState<'a> {
    row: u8,
    col: u8,
    world: &'a WindyGridworld,
    is_terminal: bool,
    wind: Option<(u8, Direction)>,
}

impl<'a> WindyGridworldState<'a> {
    pub fn new(
        row: u8,
        col: u8,
        world: &'a WindyGridworld,
        is_terminal: bool,
        wind: Option<(u8, Direction)>,
    ) -> Self {
        Self {
            row,
            col,
            world,
            is_terminal,
            wind,
        }
    }
}

impl State for WindyGridworldState<'_> {
    fn get_id(&self) -> String {
        format!("{}_{}", self.row, self.col)
    }

    fn get_actions(&self) -> Vec<String> {
        [
            Direction::North,
            Direction::East,
            Direction::South,
            Direction::West,
        ]
        .iter()
        .map(|d| d.to_string())
        .collect()
    }

    fn is_terminal(&self) -> bool {
        self.is_terminal
    }

    fn take_action(&self, action: &str) -> (f64, Self) {
        let mut new_row: i16 = self.row as i16;
        let mut new_col: i16 = self.col as i16;

        match action {
            "East" => new_col += 1,
            "West" => new_col -= 1,
            "North" => new_row += 1,
            "South" => new_row -= 1,
            _ => panic!("invalid action"),
        }

        if let Some((str, dir)) = self.wind {
            match dir {
                Direction::North => new_row += str as i16,
                Direction::East => new_col += str as i16,
                Direction::South => new_row -= str as i16,
                Direction::West => new_col -= str as i16,
            }
        }

        new_row = self.world.height.min(new_row);
        new_col = self.world.width.min(new_col);
        new_row = 0.max(new_row);
        new_col = 0.max(new_col);

        let next_state = self
            .world
            .make_state_for_row_col(new_row as u8, new_col as u8);
        let reward = if next_state.is_terminal { 1.0 } else { 0.0 };
        (reward, next_state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attempts_at_framework::v1::agent::Sarsa;
    use crate::attempts_at_framework::v1::policy::Policy;

    #[test]
    fn test_windy_gridworld() {
        let world = WindyGridworld::new(6, 9);
        let starting_point = world.make_state_for_row_col(3, 0);

        let mut agent = Sarsa::new(0.1, 0.5, 1.0);
        agent.lear_for_episode_count(200, vec![starting_point.clone()]);

        let policy = agent.get_policy();

        let mut steps: Vec<String> = Vec::new();
        let mut state = starting_point.clone();
        while !state.is_terminal() {
            let action = policy
                .select_action_for_state(&state.get_id())
                .unwrap_or("nope".to_string());
            steps.push(action.clone());
            state = state.take_action(&action).1;
        }

        println!("{:?}", steps);
    }
}
