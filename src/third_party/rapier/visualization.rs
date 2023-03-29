use crate::core::math::Vector;
use bevy_egui::{egui, EguiContext};
use na::{vector, Vector4};
use std::mem;

#[derive(Clone, Debug, PartialEq)]
pub struct VisualizationMode {
    pub particle_mode: ParticleMode,
    pub show_particles: bool,
    pub particle_scale: f32,
    pub particle_volume: bool,
    pub grid_mode: GridMode,
    pub show_grid: bool,
    pub grid_scale: f32,
    pub show_rigid_particles: bool,
    pub rigid_particle_len: usize,
    pub rigid_particle_scale: f32,
}

impl Default for VisualizationMode {
    fn default() -> Self {
        Self {
            particle_mode: ParticleMode::StaticColor,
            show_particles: true,
            particle_scale: 0.02,
            particle_volume: true,
            grid_mode: GridMode::Blocks,
            show_grid: false,
            grid_scale: 0.06,
            show_rigid_particles: false,
            rigid_particle_len: 10,
            rigid_particle_scale: 0.02,
        }
    }
}

const DEFAULT_PARTICLE_MODES: [ParticleMode; 6] = [
    ParticleMode::StaticColor,
    ParticleMode::VelocityColor {
        min: 0.0,
        max: 100.0,
    },
    ParticleMode::DensityRatio { max: 10.0 },
    ParticleMode::Position {
        mins: vector![-10.0, -10.0, -10.0],
        maxs: vector![10.0, 10.0, 10.0],
    },
    ParticleMode::Blocks { block_len: 8 },
    ParticleMode::Cdf {
        show_distance: true,
        show_normal: true,
        show_color: true,
        max_distance: 1.2,
        normal_difference: 0.5,
    },
];

const DEFAULT_GRID_MODES: [GridMode; 2] = [
    GridMode::Blocks,
    GridMode::Cdf {
        show_distance: true,
        show_color: true,
        only_show_affine: true,
        max_distance: 1.2,
    },
];

pub const COLORS: [Vector4<f32>; 6] = [
    vector![191.0 / 255.0, 57.0 / 255.0, 43.0 / 255.0, 0.0],
    vector![155.0 / 255.0, 89.0 / 255.0, 182.0 / 255.0, 0.0],
    vector![41.0 / 255.0, 128.0 / 255.0, 185.0 / 255.0, 0.0],
    vector![38.0 / 255.0, 174.0 / 255.0, 96.0 / 255.0, 0.0],
    vector![241.0 / 255.0, 196.0 / 255.0, 15.0 / 255.0, 0.0],
    vector![229.0 / 255.0, 126.0 / 255.0, 35.0 / 255.0, 0.0],
];

/// How the fluids should be rendered by the testbed.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ParticleMode {
    /// Use a plain color.
    StaticColor,
    /// Use a red taint the closer to `max` the velocity is.
    VelocityColor {
        /// Fluids with a velocity smaller than this will not have any red taint.
        min: f32,
        /// Fluids with a velocity greater than this will be completely red.
        max: f32,
    },
    DensityRatio {
        max: f32,
    },
    Position {
        mins: Vector<f32>,
        maxs: Vector<f32>,
    },
    Blocks {
        block_len: usize,
    },
    Cdf {
        show_distance: bool,
        show_normal: bool,
        show_color: bool,
        max_distance: f32,
        normal_difference: f32,
    },
}

impl ParticleMode {
    pub fn same_mode(&self, other: &Self) -> bool {
        mem::discriminant(self) == mem::discriminant(other)
    }
    pub fn name(&self) -> &str {
        match self {
            ParticleMode::StaticColor => "Static Color",
            ParticleMode::VelocityColor { .. } => "Velocity Color",
            ParticleMode::DensityRatio { .. } => "Density Ratio",
            ParticleMode::Position { .. } => "Position",
            ParticleMode::Blocks { .. } => "Blocks",
            ParticleMode::Cdf { .. } => "Cdf",
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum GridMode {
    Blocks,
    Cdf {
        show_distance: bool,
        show_color: bool,
        only_show_affine: bool,
        max_distance: f32,
    },
}

impl GridMode {
    pub fn same_mode(&self, other: &Self) -> bool {
        mem::discriminant(self) == mem::discriminant(other)
    }
    pub fn name(&self) -> &str {
        match self {
            GridMode::Blocks { .. } => "Blocks",
            GridMode::Cdf { .. } => "Cdf",
        }
    }
}

pub(crate) fn visualization_ui(mode: &mut VisualizationMode, ui_context: &EguiContext) {
    egui::Window::new("Debug Visualization")
        .anchor(egui::Align2::RIGHT_TOP, egui::vec2(0.0, 0.0))
        .show(ui_context.ctx(), |ui| {
            ui.heading("Particles");
            ui.checkbox(&mut mode.show_particles, "Show Particles");

            if mode.show_particles {
                ui.checkbox(&mut mode.particle_volume, "Use Particle Volume");
                ui.add(egui::Slider::new(&mut mode.particle_scale, 0.0..=0.1).text("Scale"));

                let particle_mode = &mut mode.particle_mode;
                egui::ComboBox::from_label("Particle Mode")
                    .selected_text(particle_mode.name())
                    .show_ui(ui, |ui| {
                        for mode in DEFAULT_PARTICLE_MODES {
                            if ui
                                .add(egui::SelectableLabel::new(
                                    particle_mode.same_mode(&mode),
                                    mode.name(),
                                ))
                                .clicked()
                            {
                                *particle_mode = mode;
                            }
                        }
                    });

                match particle_mode {
                    ParticleMode::StaticColor => {}
                    ParticleMode::DensityRatio { max } => {
                        ui.add(egui::Slider::new(max, 0.0..=10.0).text("Max Density Ratio"));
                    }
                    ParticleMode::VelocityColor { min, max } => {
                        ui.add(egui::Slider::new(min, 0.0..=10.0).text("Min Velocity"));
                        ui.add(egui::Slider::new(max, 0.0..=10.0).text("Max Velocity"));
                    }
                    ParticleMode::Position { mins, maxs } => {
                        let mut min = mins.min();
                        let mut max = maxs.max();
                        ui.add(egui::Slider::new(&mut min, -20.0..=20.0).text("Min Velocity"));
                        ui.add(egui::Slider::new(&mut max, -20.0..=20.0).text("Max Velocity"));

                        *mins = Vector::<f32>::from_element(min);
                        *maxs = Vector::<f32>::from_element(max);
                    }
                    ParticleMode::Blocks { block_len } => {
                        ui.add(egui::Slider::new(block_len, 1..=64).text("Block Length"));
                    }
                    ParticleMode::Cdf {
                        show_distance,
                        show_normal,
                        show_color,
                        max_distance,
                        normal_difference,
                    } => {
                        ui.checkbox(show_distance, "Show Distance");
                        ui.checkbox(show_normal, "Show Normal");
                        ui.checkbox(show_color, "Show Color");
                        ui.add(egui::Slider::new(max_distance, 0.001..=2.0).text("Max Distance"));
                        ui.add(
                            egui::Slider::new(normal_difference, 0.0..=1.0)
                                .text("Normal Difference"),
                        );
                    }
                }
            }

            ui.separator();
            ui.heading("Rigid Particles");
            ui.checkbox(&mut mode.show_rigid_particles, "Show Rigid Particles");

            if mode.show_rigid_particles {
                ui.add(egui::Slider::new(&mut mode.rigid_particle_scale, 0.0..=0.1).text("Scale"));
                ui.add(egui::Slider::new(&mut mode.rigid_particle_len, 1..=50).text("Length"));
            }

            ui.separator();
            ui.heading("Grid");
            ui.checkbox(&mut mode.show_grid, "Show Grid");

            if mode.show_grid {
                ui.add(egui::Slider::new(&mut mode.grid_scale, 0.0..=0.1).text("Scale"));

                let grid_mode = &mut mode.grid_mode;
                egui::ComboBox::from_label("Grid Mode")
                    .selected_text(grid_mode.name())
                    .show_ui(ui, |ui| {
                        for mode in DEFAULT_GRID_MODES {
                            if ui
                                .add(egui::SelectableLabel::new(
                                    grid_mode.same_mode(&mode),
                                    mode.name(),
                                ))
                                .clicked()
                            {
                                *grid_mode = mode;
                            }
                        }
                    });

                match grid_mode {
                    GridMode::Blocks => {}
                    GridMode::Cdf {
                        show_distance,
                        show_color,
                        only_show_affine,
                        max_distance,
                    } => {
                        ui.checkbox(show_distance, "Show Distance");
                        ui.checkbox(show_color, "Show Color");
                        ui.checkbox(only_show_affine, "Only show affine");
                        ui.add(egui::Slider::new(max_distance, 0.001..=2.0).text("Max Distance"));
                    }
                }
            }
        });
}
