pub fn linear_differentiable_function(values: &Vec<f64>, weights: &Vec<f64>) -> f64 {
    assert_eq!(values.len(), weights.len());
    (0..values.len()).fold(0.0, |acc, index| {
        acc + (values[index] * weights[index])
    })
}

pub fn weight_update(values: &Vec<f64>, weights: &Vec<f64>, learning_rate: f64, expected_value: f64) -> Vec<f64> {
    assert_eq!(values.len(), weights.len());
    let current_value = linear_differentiable_function(values, weights);
    let error = expected_value - current_value;
    let gradient = error * learning_rate;
    values.iter().enumerate().map(|(index, value)| {
        weights[index] + (gradient * value)
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_case() {
        let values = vec![1.0, 2.0, 3.0];
        let weights = vec![1.0, 2.0, 3.0];
        let result = linear_differentiable_function(&values, &weights);
        assert_eq!(result, 14.0);
    }

    #[test]
    fn weight_update_test() {
        let learning_rate = 0.1;
        let values = vec![1.0, 2.0];
        let mut weights = vec![1.0, 1.0];
        let expected_state_value = 10.0;

        let mut current_value = linear_differentiable_function(&values, &weights);
        let mut count = 0;

        while (current_value - expected_state_value).abs() > 0.00001 {
            if count > 1000 {
                panic!("Failed to converge");
            }
            weights = weight_update(&values, &weights, learning_rate, expected_state_value);
            current_value = linear_differentiable_function(&values, &weights);
            count += 1;
        }

        println!("weights: {:?}, converged after: {} iterations", weights, count);
    }
}