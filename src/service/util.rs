/// Calculates the new average given the current average, total count of items,
/// and a newly added reward.
///
/// This function is based on an incremental formula to calculate the average without 
/// needing to sum all the previously added values. It updates the current average 
/// by taking into account the new reward and the total number of items processed so far.
///
/// # Arguments
///
/// * `current_average` - The current average value computed up to now.
/// * `total_count` - The total number of items, including the new item.
/// * `new_reward` - The new reward value to be added.
///
/// # Returns
///
/// Returns the updated average as a `f64`.
///
/// # Examples
///
/// ```
/// use ReinforcementLearning::service::calc_average;
///
/// let numbers = vec![3.0, 3.6, 4.1, 5.3];
/// let o_n_average: f64 = numbers.iter().sum::<f64>() / numbers.len() as f64;
///
/// let o_1_average: f64 = numbers.iter().enumerate().fold(0.0, |acc, (i, next)| {
///     calc_average(acc, (i + 1) as i32, *next)
/// });
///
/// assert_eq!(o_1_average, o_n_average);
/// ```
pub fn calc_average(
    current_average: f64,
    total_count: i32,
    new_reward: f64,
) -> f64 {
    current_average + (1.0 / total_count as f64) * (new_reward - current_average)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calc_average() {
        let numbers = vec![1.0, -1.0];
        let o_n_average: f64 = numbers.iter().sum::<f64>() / numbers.len() as f64;

        let o_1_average: f64 = numbers.iter().enumerate().fold(0.0, |acc, (i, next)| {
            calc_average(acc, (i + 1) as i32, *next)
        });
        assert_eq!(o_1_average, o_n_average);
    }
}