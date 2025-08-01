pub mod policy_gradient;

pub fn soft_max(preferences: &[f64]) -> Vec<f64> {
    let max_pref = preferences
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let exp_prefs: Vec<f64> = preferences.iter().map(|x| (x - max_pref).exp()).collect();
    let denominator = exp_prefs.iter().sum::<f64>();
    exp_prefs.iter().map(|x| x / denominator).collect()
}
