pub mod attempts_at_framework;
mod chapter_02;
mod chapter_03;
mod chapter_04;
pub mod chapter_05;
pub mod chapter_06;
pub mod chapter_07;
pub mod chapter_08;
mod chapter_09;
mod chapter_10;
mod chapter_11;
mod chapter_12;
pub mod chess_state;
pub mod chess_state_v2;
pub mod service;

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
