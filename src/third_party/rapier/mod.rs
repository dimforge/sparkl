//! Two-way coupling with the Rapier physics engine.
pub(self) mod point_cloud_render;
mod testbed_plugin;
mod visualization;

pub use testbed_plugin::{MpmTestbedPlugin, UserCallback};
