
fn iterative_policy_evaluation(states: Vec<f32>, theta: f32) -> f32 {
    loop {
        let mut delta: f32 = 0_f32;
        states.iter().for_each(|state| {
            let v = state;


        });
        if delta < theta {
            break;
        }
    }
    0.0
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_iterative_policy_evaluation() {

    }
}