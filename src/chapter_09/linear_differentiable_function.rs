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
    use std::path::PathBuf;
    use plotters::prelude::ShapeStyle;
    use plotters::style::BLUE;
    use crate::service::{LineChartBuilder, LineChartData};
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

    #[test]
    fn graph_learning_rates() {
        let learning_rates = vec![0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0];

        let initial_weights = vec![1.0, 1.0];
        let values = vec![1.0, 2.0];
        let expected_state_value = -10.0;

        let mut chart_builder = LineChartBuilder::new();
        chart_builder
            .set_path(PathBuf::from("output/chapter9/learning_rates_and_linear_regression_convergence.png"))
            .set_title("Learning rates vs iterations to convergence".to_string())
            .set_x_label("Learning rate".to_string())
            .set_y_label("Iterations to convergence".to_string());

        let mut data: Vec<(f32, f32)> = Vec::new();

        for learning_rate in learning_rates {
            let mut weights = initial_weights.clone();
            let mut current_value = linear_differentiable_function(&values, &weights);
            let mut count = 0;

            while (current_value - expected_state_value).abs() > 0.00001 {
                if count > 1000 {
                    break;
                }
                weights = weight_update(&values, &weights, learning_rate, expected_state_value);
                current_value = linear_differentiable_function(&values, &weights);
                count += 1;
            }

            println!("learning rate: {}, weights: {:?}, converged after: {} iterations",learning_rate, weights, count);

            data.push((learning_rate as f32, count as f32));
        }

        let data_as_line = LineChartData::new("idk".to_string(), data, ShapeStyle::from(&BLUE));
        chart_builder.add_data(data_as_line);

        chart_builder.create_chart().unwrap()
    }
}