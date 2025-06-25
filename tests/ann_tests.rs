use std::fmt::format;
use ReinforcementLearning::attempts_at_framework::v2::artificial_neural_network::loss_functions::LossFunction;
use ReinforcementLearning::attempts_at_framework::v2::artificial_neural_network::loss_functions::mean_squared_error::MeanSquaredError;
use ReinforcementLearning::attempts_at_framework::v2::artificial_neural_network::model::model_builder::{LayerBuilder, ModelBuilder};
use ReinforcementLearning::attempts_at_framework::v2::artificial_neural_network::model::model_builder::LayerType::{LINEAR, RELU};
use std::fs::File;
use std::hash::Hash;
use std::io::{read_to_string, BufRead, BufReader, Read};
use egui::ahash::HashMap;
use rand::prelude::SliceRandom;
use serde_json::Value;

#[test]
fn test_ann_2x_plus_y() {
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

#[test]
fn test_job_ad_identification() {
    let file =
        File::open("tests/job_ad_data/word_index.json").expect("Failed to open word list file");
    let reader = BufReader::new(file);
    let text = read_to_string(reader).expect("Failed to read word list file");
    let data: Value = serde_json::from_str(&text).expect("Failed to parse word list file");

    let word_indexes = data
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.to_string(), v.as_u64().unwrap()))
        .collect::<HashMap<String, u64>>();

    let mut model_builder = ModelBuilder::new();
    model_builder.add_layer(LayerBuilder::new(LINEAR, 1)); // really this should be a sigmoid, but I have not crated that yet. let's see how well a simple linear layer does.
    let input_size = word_indexes.len();
    model_builder.set_input_size(input_size);
    model_builder.set_loss_function(Box::new(MeanSquaredError));
    let mut model = model_builder.build().unwrap();

    let positive_cases = File::open("tests/job_ad_data/job_postings_text.csv")
        .expect("Failed to open positive cases file");
    let reader = BufReader::new(positive_cases);
    let positive_data_points = read_to_string(reader).expect("Failed to read positive cases file");
    let positive_data_points = positive_data_points.split("\n").collect::<Vec<&str>>();
    let mut test_cases: Vec<(Vec<f64>, f64)> = positive_data_points
        .iter()
        .map(|x| {
            let data = x
                .split(",")
                .map(|d| match d.trim() {
                    "0" => 0.0,
                    "1" => 1.0,
                    _ => 0.0,
                })
                .collect::<Vec<f64>>();
            (data, 1.0)
        })
        .collect::<Vec<(Vec<f64>, f64)>>();

    let negative_cases = File::open("tests/job_ad_data/non_job_postings_text.csv")
        .expect("Failed to open negative cases file");
    let reader = BufReader::new(negative_cases);
    let negative_data_points = read_to_string(reader).expect("Failed to read negative cases file");
    let negative_data_points = negative_data_points.split("\n").collect::<Vec<&str>>();
    let mut negative_cases: Vec<(Vec<f64>, f64)> = negative_data_points
        .iter()
        .map(|x| {
            let data = x
                .split(",")
                .map(|d| match d.trim() {
                    "0" => 0.0,
                    "1" => 1.0,
                    _ => 0.0,
                })
                .collect::<Vec<f64>>();
            (data, 0.0)
        })
        .collect::<Vec<(Vec<f64>, f64)>>();

    test_cases.append(&mut negative_cases);
    let mut filtered_test_cases: Vec<&(Vec<f64>, f64)> = test_cases
        .iter()
        .filter(|x| x.0.len() == input_size)
        .collect();
    filtered_test_cases.shuffle(&mut rand::rng());

    let learning_rate = 0.01;
    let delta = 0.1;
    let epochs = 100;
    let loss_function = MeanSquaredError;
    for epoch in 0..epochs {
        let mut total_error = 0.0;
        for (input, expected) in &filtered_test_cases {
            model.train(input.clone(), vec![expected.clone()], learning_rate);

            let predicted = model.predict(input.clone());
            let error = loss_function.calculate_loss(&vec![expected.clone()], &predicted);
            total_error += error;
        }

        println!("Epoch: {} Error: {}", epoch, total_error);

        if total_error < delta {
            println!("Epoch: {} Error: {}", epoch, total_error);
            break;
        }
    }

    let probably_a_job_ad_text = "Senior Software Engineer";
    let probably_not_a_job_ad_text = "Get yours today";

    let mut probably_a_job_ad_input = vec![0.0; input_size];
    probably_a_job_ad_text
        .to_lowercase()
        .split(" ")
        .for_each(|x| {
            let index = match word_indexes.get(x) {
                None => 0,
                Some(&v) => v as usize,
            };
            probably_a_job_ad_input[index] = 1.0;
        });
    let mut probably_not_a_job_ad_input = vec![0.0; input_size];
    probably_not_a_job_ad_text
        .to_lowercase()
        .split(" ")
        .for_each(|x| {
            let index = match word_indexes.get(x) {
                None => 0,
                Some(&v) => v as usize,
            };
            probably_not_a_job_ad_input[index] = 1.0;
        });

    let probably_a_job_ad_output = model.predict(probably_a_job_ad_input);
    let probably_not_a_job_ad_output = model.predict(probably_not_a_job_ad_input);

    println!("Probably a job ad: {:?}", probably_a_job_ad_output);
    println!("Probably not a job ad: {:?}", probably_not_a_job_ad_output);
}
