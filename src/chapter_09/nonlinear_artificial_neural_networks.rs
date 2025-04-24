// fn activate(&self, inputs: &[f64]) -> f64 {
//     let weighted_inputs = inputs
//         .iter()
//         .zip(self.weights.iter())
//         .map(|(&x, &w)| x * w)
//         .sum::<f64>()
//         + self.bias;
//
//     1.0 / ((-weighted_inputs).exp() + 1.0)
// }
