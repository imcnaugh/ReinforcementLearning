use plotters::prelude::*;

fn wat(plots: Vec<(f32, f32)>) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("output/my_chart.png", (640, 480)).into_drawing_area();

    std::fs::create_dir_all("output")?;

    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("y=x^2", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(-1f32..1f32, -0.1f32..1f32)?;

    chart.configure_mesh().draw()?;

    chart
        .draw_series(LineSeries::new(
            plots,
            &BLUE,
        ))?
        .label("y = x")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wat_that_thang_do() {
        let idk = vec![(0.0, 0.0), (0.3,0.2), (0.5, 0.9)];

        match wat(idk) {
            Ok(_) => {assert!(true, "wat")}
            Err(e) => {panic!("wat {}", e)}
        }
    }
}