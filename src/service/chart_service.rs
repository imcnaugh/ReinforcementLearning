use plotters::prelude::*;
use std::error::Error;
use std::path::PathBuf;

pub struct LineChartData {
    label: Option<String>,
    points: Vec<(f32, f32)>,
    style: Option<ShapeStyle>,
}

impl LineChartData {
    pub fn new_with_style(
        label: String,
        points: Vec<(f32, f32)>,
        style: ShapeStyle,
    ) -> LineChartData {
        LineChartData {
            label: Some(label),
            points,
            style: Some(style),
        }
    }

    pub fn new(label: String, points: Vec<(f32, f32)>) -> LineChartData {
        LineChartData {
            label: Some(label),
            points,
            style: None,
        }
    }

    pub fn just_plot_it(points: Vec<(f32, f32)>) -> LineChartData {
        Self {
            label: None,
            points,
            style: None,
        }
    }
}

pub struct LineChartBuilder {
    title: Option<String>,
    x_label: Option<String>,
    y_label: Option<String>,
    output_path: Option<PathBuf>,
    png_size: Option<(u32, u32)>,
    data: Vec<LineChartData>,
    graph_margin: Option<f32>,
}

impl LineChartBuilder {
    pub fn new() -> LineChartBuilder {
        LineChartBuilder {
            title: None,
            x_label: None,
            y_label: None,
            output_path: None,
            png_size: None,
            data: vec![],
            graph_margin: None,
        }
    }

    pub fn set_path(&mut self, path: PathBuf) -> &mut LineChartBuilder {
        self.output_path = Some(path);
        self
    }

    pub fn set_title(&mut self, title: String) -> &mut LineChartBuilder {
        self.title = Some(title);
        self
    }

    pub fn set_x_label(&mut self, label: String) -> &mut LineChartBuilder {
        self.x_label = Some(label);
        self
    }

    pub fn set_y_label(&mut self, label: String) -> &mut LineChartBuilder {
        self.y_label = Some(label);
        self
    }

    pub fn set_size(&mut self, width: u32, height: u32) -> &mut LineChartBuilder {
        self.png_size = Some((width, height));
        self
    }

    pub fn add_data(&mut self, plot: LineChartData) -> &mut LineChartBuilder {
        self.data.push(plot);
        self
    }

    pub fn add_graph_margin(&mut self, margin: f32) -> &mut LineChartBuilder {
        self.graph_margin = Some(margin);
        self
    }

    pub fn create_chart(self) -> Result<(), Box<dyn Error>> {
        let size = self.png_size.unwrap_or((1200, 900));
        let path = &Self::get_output_path(self.output_path)?;
        let root = BitMapBackend::new(path, size).into_drawing_area();

        root.fill(&WHITE)?;

        let mut builder = ChartBuilder::on(&root);
        let builder = builder
            .margin(5)
            .x_label_area_size(60)
            .y_label_area_size(60);
        let builder = match self.title {
            None => builder,
            Some(title) => builder.caption(title, ("sans-serif", 60).into_font()),
        };

        let (x_min, x_max, y_min, y_max) = &Self::get_graph_bounds(&self.data);
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
            configure_mesh
                .x_desc(x_desc)
                .label_style(("sans-serif", 30).into_font());
        }
        if let Some(y_desc) = self.y_label {
            configure_mesh
                .y_desc(y_desc)
                .label_style(("sans-serif", 30).into_font());
        }
        configure_mesh
            .x_label_style(("sans-serif", 20).into_font())
            .y_label_style(("sans-serif", 20).into_font());

        configure_mesh.draw()?;

        let default_styles: Vec<ShapeStyle> = vec![
            ShapeStyle::from(&RED),
            ShapeStyle::from(&BLUE),
            ShapeStyle::from(&MAGENTA),
            ShapeStyle::from(&CYAN),
            ShapeStyle::from(&GREEN),
            ShapeStyle::from(&YELLOW),
        ];
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
                .legend(move |(x, y)| {
                    PathElement::new(vec![(x, y), (x + 20, y)], style.stroke_width(3))
                });
        });

        chart
            .configure_series_labels()
            .label_font(("sans-sarif", 30).into_font())
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()?;

        root.present()?;

        Ok(())
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

    fn get_graph_bounds(data_points: &Vec<LineChartData>) -> (f32, f32, f32, f32) {
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
}

pub struct MultiLineChartData {
    label: Option<String>,
    points: Vec<f64>,
}

impl MultiLineChartData {
    pub fn new(points: Vec<f64>) -> MultiLineChartData {
        MultiLineChartData {
            label: None,
            points,
        }
    }

    pub fn set_label(&mut self, label: String) {
        self.label = Some(label);
    }
}

pub struct MultiLineChartBuilder {
    title: Option<String>,
    output_path: Option<PathBuf>,
    png_size: Option<(u32, u32)>,
    data: Vec<MultiLineChartData>,
}

impl MultiLineChartBuilder {
    pub fn new() -> MultiLineChartBuilder {
        MultiLineChartBuilder {
            title: None,
            output_path: None,
            png_size: None,
            data: vec![],
        }
    }

    pub fn set_path(&mut self, path: PathBuf) -> &mut MultiLineChartBuilder {
        self.output_path = Some(path);
        self
    }

    pub fn set_title(&mut self, title: String) -> &mut MultiLineChartBuilder {
        self.title = Some(title);
        self
    }

    pub fn set_size(&mut self, width: u32, height: u32) -> &mut MultiLineChartBuilder {
        self.png_size = Some((width, height));
        self
    }

    pub fn add_data(&mut self, plot: MultiLineChartData) -> &mut MultiLineChartBuilder {
        self.data.push(plot);
        self
    }

    pub fn create_chart(self) -> Result<(), Box<dyn Error>> {
        let path = &Self::get_output_path(self.output_path).unwrap();
        let size = self.png_size.unwrap_or((1200, 900));
        let area = SVGBackend::new(path, size).into_drawing_area();

        area.fill(&WHITE).unwrap();

        let max_width = self.data.iter().map(|d| d.points.len()).max().unwrap() - 1;

        let max_element = self
            .data
            .iter()
            .flat_map(|d| d.points.iter())
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        let max_depth = self.data.len() - 1;

        let x_axis = (0.0..max_width as f64).step(1.0);
        // let y_axis = (0.0..max_element).step(1.0);
        let y_axis = (-1.0..1.0).step(0.1);
        let z_axis = (max_depth as f64..0.0).step(-1.0);

        let mut chart = ChartBuilder::on(&area)
            .caption("3D Plot Test", ("sans", 20))
            .build_cartesian_3d(x_axis.clone(), y_axis.clone(), z_axis.clone())
            .unwrap();

        chart.with_projection(|mut pb| {
            pb.yaw = 0.5;
            pb.scale = 0.9;
            pb.into_matrix()
        });

        chart
            .configure_axes()
            .light_grid_style(BLACK.mix(0.15))
            .max_light_lines(3)
            .draw()
            .unwrap();

        let mut styles = vec![BLUE, RED, GREEN, BLACK].into_iter().cycle();

        self.data.iter().enumerate().for_each(|(i, d)| {
            let line_data: Vec<(f64, f64, f64)> = d
                .points
                .iter()
                .enumerate()
                .map(|(x, y)| (x as f64, y.clone(), i as f64))
                .collect();
            let style = styles.next().unwrap();
            chart
                .draw_series(LineSeries::new(line_data, style.clone()))
                .unwrap()
                .label(d.label.clone().unwrap_or_else(|| i.to_string()))
                .legend(move |(x, y)| {
                    PathElement::new(
                        vec![(x, y), (x + 20, y)],
                        ShapeStyle::from(style.clone()).stroke_width(2),
                    )
                });
        });

        chart
            .configure_series_labels()
            .border_style(BLACK)
            .draw()
            .unwrap();

        // To avoid the IO failure being ignored silently, we manually call the present function
        area.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");

        Ok(())
    }

    fn get_output_path(output_path: Option<PathBuf>) -> Result<PathBuf, Box<dyn Error>> {
        let output_path = output_path.unwrap_or_else(|| "multiLineChart.png".into());
        if let Some(parent) = output_path.parent() {
            if !parent.exists() {
                std::fs::create_dir_all(parent)?;
            }
        }
        Ok(output_path)
    }
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
        let data = LineChartData::just_plot_it(points);

        let data_2 = LineChartData::new_with_style(
            String::from("Data 2"),
            points_2,
            ShapeStyle::from(&BLACK),
        );

        let data_3 = LineChartData::just_plot_it(points_3);

        let mut builder = LineChartBuilder::new();
        builder
            .set_path(PathBuf::from("output/my_chart.png"))
            .add_data(data)
            .set_title(String::from("Some Title"))
            .set_x_label(String::from("The X axis"))
            .set_y_label(String::from("The Y axis"))
            // .set_size(1200, 900)
            .add_data(data_2)
            .add_data(data_3);
        builder.create_chart().expect("TODO: panic message");
    }

    #[test]
    fn plot_3d_graph_test() {
        let data = vec![1_f64, 1.2, 1.5, 1.2];
        let data_2 = data.clone().iter().map(|x| x + 0.3).collect();

        let d_1 = MultiLineChartData::new(data);
        let d_2 = MultiLineChartData::new(data_2);

        let mut builder = MultiLineChartBuilder::new();
        builder
            .set_path(PathBuf::from("output/multiLineChart.png"))
            .add_data(d_1)
            .add_data(d_2);
        builder.create_chart().expect("TODO: panic message");
    }
}
