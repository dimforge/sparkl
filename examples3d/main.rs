#![allow(dead_code)]

extern crate nalgebra as na;

mod cube_through_sand3;
mod cutting_sand3;
mod fluids3;
mod helper;
mod sand3;
mod sand_penetration3;
mod wheel3;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use inflector::Inflector;
use rapier_testbed3d::{Testbed, TestbedApp};
use sparkl3d::third_party::rapier::MpmTestbedPlugin;
use std::cmp::Ordering;

fn demo_name_from_command_line() -> Option<String> {
    let mut args = std::env::args();

    while let Some(arg) = args.next() {
        if &arg[..] == "--example" {
            return args.next();
        }
    }

    None
}

#[cfg(any(target_arch = "wasm32", target_arch = "asmjs"))]
fn demo_name_from_url() -> Option<String> {
    None
    //    let window = stdweb::web::window();
    //    let hash = window.location()?.search().ok()?;
    //    if hash.len() > 0 {
    //        Some(hash[1..].to_string())
    //    } else {
    //        None
    //    }
}

#[cfg(not(any(target_arch = "wasm32", target_arch = "asmjs")))]
fn demo_name_from_url() -> Option<String> {
    None
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub fn main() {
    // env_logger::init();
    let demo = demo_name_from_command_line()
        .or_else(|| demo_name_from_url())
        .unwrap_or(String::new())
        .to_camel_case();

    let mut builders: Vec<(_, fn(&mut Testbed))> = vec![
        ("Cube through sand", cube_through_sand3::init_world),
        ("Cutting sand", cutting_sand3::init_world),
        ("Sand penetration", sand_penetration3::init_world),
        ("Wheel", wheel3::init_world),
        ("Elasticity", sand3::init_world),
        ("Fluids", fluids3::init_world),
    ];

    // Lexicographic sort, with stress tests moved at the end of the list.
    builders.sort_by(|a, b| match (a.0.starts_with("("), b.0.starts_with("(")) {
        (true, true) | (false, false) => a.0.cmp(b.0),
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
    });

    let i = builders
        .iter()
        .position(|builder| builder.0.to_camel_case().as_str() == demo.as_str())
        .unwrap_or(0);

    let testbed = TestbedApp::from_builders(i, builders);
    testbed.run_with_init(MpmTestbedPlugin::init)
}
