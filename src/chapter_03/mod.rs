
#[cfg(test)]
mod tests {
    #[test]
    fn exercise_3_10() {
        const DISCOUNT_RATE: f32 = 0.5f32;

        let a: f32 = (0..100).map(|i| DISCOUNT_RATE.powi(i)).sum();
        let b = 1f32 / (1f32 - DISCOUNT_RATE);

        println!("a: {}, b: {}", a, b);
        assert_eq!(a, b);
    }
}