use crate::attempts_at_framework::v2::state::State;

#[derive(Clone)]
struct TestState {
    id: usize,
}

impl TestState {
    fn new(id: usize) -> Self {
        let id = id.clamp(1, 7);
        Self { id }
    }
}

impl State for TestState {
    fn get_id(&self) -> String {
        self.id.to_string()
    }

    fn get_actions(&self) -> Vec<String> {
        todo!()
    }

    fn is_terminal(&self) -> bool {
        false
    }

    fn take_action(&self, action: &str) -> (f64, Self) {
        todo!()
    }

    fn get_values(&self) -> Vec<f64> {
        let mut values = vec![0.0; 8];
        let index = self.id - 1;
        values[index] = 2.0;
        values[7] = 1.0;
        values
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_values() {
        (1..=7).for_each(|id| {
            let state = TestState::new(id);
            let values = state.get_values();
            let mut expected_values = vec![0.0; 8];
            expected_values[id - 1] = 2.0;
            expected_values[7] = 1.0;
            assert_eq!(values, expected_values);
        });
    }
}
