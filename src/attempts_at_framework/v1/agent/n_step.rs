use crate::attempts_at_framework::v1::policy::EGreedyPolicy;

pub struct NStep {
    n: usize,
    policy: EGreedyPolicy,
    default_action_value: f64,
    step_size_parameter: f64,
    discount_factor: f64,
}

impl NStep {
    pub fn new(n: usize, e: f64, step_size_parameter: f64, discount_factor: f64) -> Self {
        Self {
            n,
            policy: EGreedyPolicy::new(e),
            default_action_value: 0.0,
            step_size_parameter,
            discount_factor,
        }
    }
}
