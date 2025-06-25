use ReinforcementLearning::attempts_at_framework::v2::artificial_neural_network::loss_functions::LossFunction;
use ReinforcementLearning::attempts_at_framework::v2::artificial_neural_network::loss_functions::mean_squared_error::MeanSquaredError;
use ReinforcementLearning::attempts_at_framework::v2::artificial_neural_network::model::model_builder::{LayerBuilder, ModelBuilder};
use ReinforcementLearning::attempts_at_framework::v2::artificial_neural_network::model::model_builder::LayerType::{LINEAR, RELU};

#[test]
fn test_ann() {
    let mut model_builder = ModelBuilder::new();
    model_builder.add_layer(LayerBuilder::new(RELU, 2));
    model_builder.add_layer(LayerBuilder::new(LINEAR, 1));
    model_builder.set_input_size(2);
    model_builder.set_loss_function(Box::new(MeanSquaredError));
    let mut model = model_builder.build().unwrap();

    let inputs = vec![
        vec![2.0, 3.0],
        vec![-1.0, 2.0],
        vec![1.0, 1.0],
        vec![0.0, 0.0],
        vec![-10.0, -2.0],
    ];
    let expected = vec![vec![7.0], vec![0.0], vec![3.0], vec![0.0], vec![0.0]];
    let loss_function = MeanSquaredError;

    let learning_rate = 0.01;

    let delta = 0.0001;

    for episode_count in 0..1000 {
        let mut total_error = 0.0;
        for (expected, input) in expected.iter().zip(inputs.iter()) {
            model.train(input.to_vec(), expected.to_vec(), learning_rate);
            let predicted = model.predict(input.to_vec());
            let error = loss_function.calculate_loss(expected, &predicted);
            total_error += error;
        }
        if total_error < delta {
            println!("Episode: {} Error: {}", episode_count, total_error);
            break;
        }
    }

    model.print_weights();

    let test_input = vec![1.0, 2.0];
    let test_output = model.predict(test_input);
    println!("Test output: {:?}, should be 4", test_output);
}
