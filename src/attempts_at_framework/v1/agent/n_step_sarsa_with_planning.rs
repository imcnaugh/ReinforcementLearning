use crate::attempts_at_framework::v1::policy::EGreedyPolicy;
use crate::attempts_at_framework::v1::state::State;
use std::collections::HashMap;

struct NStepSarsaWithPlanning {
    n: usize,
    policy: EGreedyPolicy,
    default_value: f64,
    step_size: f64,
    discount: f64,
    values: HashMap<String, f64>,
    total_episodes: usize,
}

impl NStepSarsaWithPlanning {
    fn new(builder: &NStepSarsaWithPlanningBuilder) -> Self {
        Self {
            n: builder.n,
            policy: EGreedyPolicy::new(builder.explore_factor),
            default_value: builder.default_value,
            step_size: builder.step_size_parameter,
            discount: builder.discount_rate,
            values: HashMap::new(),
            total_episodes: 0,
        }
    }

    pub fn get_policy(&self) -> &EGreedyPolicy {
        &self.policy
    }

    pub fn get_total_episodes(&self) -> usize {
        self.total_episodes
    }

    pub fn learn_for_episode<S: State>(&mut self, starting_state: S) {}
}

pub struct NStepSarsaWithPlanningBuilder {
    n: usize,
    explore_factor: f64,
    default_value: f64,
    step_size_parameter: f64,
    discount_rate: f64,
}

impl NStepSarsaWithPlanningBuilder {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            explore_factor: 0.1,
            default_value: 0.0,
            step_size_parameter: 0.1,
            discount_rate: 1.0,
        }
    }

    pub fn set_explore_factor(mut self, explore_factor: f64) -> Self {
        if explore_factor < 0.0 || explore_factor > 1.0 {
            panic!("Explore factor must be between 0.0 and 1.0")
        }
        self.explore_factor = explore_factor;
        self
    }

    pub fn set_default_state_value(mut self, default_state_value: f64) -> Self {
        self.default_value = default_state_value;
        self
    }

    pub fn set_step_size_parameter(mut self, step_size_parameter: f64) -> Self {
        self.step_size_parameter = step_size_parameter;
        self
    }

    pub fn set_discount_rate(mut self, discount_rate: f64) -> Self {
        if discount_rate < 0.0 || discount_rate > 1.0 {
            panic!("Discount rate must be between 0.0 and 1.0")
        }
        self.discount_rate = discount_rate;
        self
    }

    pub fn build(&self) -> NStepSarsaWithPlanning {
        NStepSarsaWithPlanning::new(self)
    }
}
