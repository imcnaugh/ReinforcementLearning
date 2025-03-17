mod stochastic;

trait Policy {
    fn pick_action_for_state(&self, state_id: &str) -> Result<&str, String>;

    fn get_actions_for_state(&self, state_id: &str) -> Result<&Vec<(f64, String)>, String>;
}
