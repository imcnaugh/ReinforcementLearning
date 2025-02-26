use plotters::prelude::*;
use std::error::Error;
use std::path::PathBuf;

pub struct ChartData {
    label: Option<String>,
    points: Vec<(f32, f32)>,
    style: Option<ShapeStyle>,
}

impl ChartData {
    pub fn new(label: String, points: Vec<(f32, f32)>, style: ShapeStyle) -> ChartData {
        ChartData {
            label: Some(label),
            points,
            style: Some(style),
        }
    }

    pub fn just_plot_it(points: Vec<(f32, f32)>) -> ChartData {
        Self {
            label: None,
            points,
            style: None,
        }
    }
}

pub struct ChartBuilder {
    title: Option<String>,
    x_label: Option<String>,
    y_label: Option<String>,
    output_path: Option<PathBuf>,
    png_size: Option<(u32, u32)>,
    data: Vec<ChartData>,
    graph_margin: Option<f32>,
}

impl ChartBuilder {
    pub fn new() -> ChartBuilder {
        ChartBuilder {
            title: None,
            x_label: None,
            y_label: None,
            output_path: None,
            png_size: None,
            data: vec![],
            graph_margin: None,
        }
    }

    pub fn set_path(&mut self, path: PathBuf) -> &mut ChartBuilder {
        self.output_path = Some(path);
        self
    }

    pub fn set_title(&mut self, title: String) -> &mut ChartBuilder {
        self.title = Some(title);
        self
    }

    pub fn set_x_label(&mut self, label: String) -> &mut ChartBuilder {
        self.x_label = Some(label);
        self
    }

    pub fn set_y_label(&mut self, label: String) -> &mut ChartBuilder {
        self.y_label = Some(label);
        self
    }

    pub fn set_size(&mut self, width: u32, height: u32) -> &mut ChartBuilder {
        self.png_size = Some((width, height));
        self
    }

    pub fn add_data(&mut self, plot: ChartData) -> &mut ChartBuilder {
        self.data.push(plot);
        self
    }

    pub fn add_graph_margin(&mut self, margin: f32) -> &mut ChartBuilder {
        self.graph_margin = Some(margin);
        self
    }

    pub fn create_chart(self) -> Result<(), Box<dyn Error>> {
        let size = self.png_size.unwrap_or((640, 480));
        let path = &get_output_path(self.output_path)?;
        let root = BitMapBackend::new(path, size).into_drawing_area();

        root.fill(&WHITE)?;

        let mut builder = plotters::prelude::ChartBuilder::on(&root);
        let mut builder = builder
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(30);
        let mut builder = match self.title {
            None => builder,
            Some(title) => builder.caption(title, ("sans-serif", 50).into_font()),
        };

        let (x_min, x_max, y_min, y_max) = get_graph_bounds(&self.data);
        let margin = self.graph_margin.unwrap_or(0.1);
        let x_dif = x_max - x_min;
        let y_dif = y_max - y_min;
        let x_min = x_min - margin * x_dif;
        let x_max = x_max + margin * x_dif;
        let y_min = y_min - margin * y_dif;
        let y_max = y_max + margin * y_dif;

        let mut chart = builder.build_cartesian_2d(x_min..x_max, y_min..y_max)?;

        let mut configure_mesh = chart.configure_mesh();
        if let Some(x_desc) = self.x_label {
            configure_mesh.x_desc(x_desc);
        }
        if let Some(y_desc) = self.y_label {
            configure_mesh.y_desc(y_desc);
        }
        configure_mesh.draw()?;

        let default_styles: Vec<ShapeStyle> = vec![BLUE.into(), RED.into(), GREEN.into()];
        let mut next_style = default_styles.iter().cycle();
        let mut next_id = 1;

        self.data.iter().for_each(|p| {
            let style = match &p.style {
                None => *next_style.next().unwrap(),
                Some(s) => *s,
            };
            let label = match &p.label {
                None => {
                    let id = next_id.clone().to_string();
                    next_id = next_id + 1;
                    id
                }
                Some(label) => label.clone(),
            };

            chart
                .draw_series(LineSeries::new(p.points.clone(), style))
                .unwrap()
                .label(label)
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], style));
        });

        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()?;

        root.present()?;

        Ok(())
    }
}

fn get_output_path(output_path: Option<PathBuf>) -> Result<PathBuf, Box<dyn Error>> {
    let output_path = output_path.unwrap_or_else(|| "chart.png".into());
    if let Some(parent) = output_path.parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent)?;
        }
    }
    Ok(output_path)
}

fn get_graph_bounds(data_points: &Vec<ChartData>) -> (f32, f32, f32, f32) {
    data_points.iter().fold(
        (
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ),
        |(min_x, max_x, min_y, max_y), data| {
            data.points.iter().fold(
                (min_x, max_x, min_y, max_y),
                |(cur_min_x, cur_max_x, cur_min_y, cur_max_y), &(x, y)| {
                    (
                        cur_min_x.min(x),
                        cur_max_x.max(x),
                        cur_min_y.min(y),
                        cur_max_y.max(y),
                    )
                },
            )
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn what_that_struct_do() {
        let points = vec![(0.0, 0.0), (1.0, 1.0), (0.0, 0.5)];
        let points_2 = vec![(-0.5, 0.0), (-1.0, 0.3), (0.0, 0.5)];
        let points_3 = points_2.iter().map(|(a, b)| (a + 0.3, b - 0.1)).collect();

        // let data = ChartData::new(
        //     String::from("Test Data"),
        //     points,
        //     ShapeStyle::from(&GREEN)
        // );
        let data = ChartData::just_plot_it(points);

        let data_2 = ChartData::new(String::from("Data 2"), points_2, ShapeStyle::from(&BLACK));

        let data_3 = ChartData::just_plot_it(points_3);

        let mut builder = ChartBuilder::new();
        builder
            .set_path(PathBuf::from("output/my_chart.png"))
            .add_data(data)
            .set_title(String::from("Some Title"))
            .set_x_label(String::from("The X axis"))
            .set_y_label(String::from("The Y axis"))
            // .set_size(900, 800)
            .add_data(data_2)
            .add_data(data_3);
        builder.create_chart();
    }
}
