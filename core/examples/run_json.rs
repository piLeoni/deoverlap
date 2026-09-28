//! Run deoverlap on polylines read from JSON; used by `tools/compare.py`.
//!
//! Input: `{"options": {...}, "paths": [[[x, y], ...], ...]}`
//! Output: `{"kept": [[[x, y], ...], ...], "removed_length": f, "wholly_removed": n, "seconds": f}`

use std::time::Instant;

use deoverlap_core::{deoverlap, geom_length, ClipMode, KeepPolicy, Options, Part};
use geo::LineString;
use serde::Deserialize;

#[derive(Deserialize)]
struct Input {
    options: Opts,
    paths: Vec<Vec<[f64; 2]>>,
}

#[derive(Deserialize)]
struct Opts {
    tolerance: f64,
    #[serde(default)]
    keep: Option<String>,
    #[serde(default)]
    mode: Option<String>,
    #[serde(default)]
    min_length: f64,
    #[serde(default)]
    parallel_only: bool,
    #[serde(default)]
    parallel_angle: Option<f64>,
    #[serde(default)]
    segments: bool,
    #[serde(default)]
    keep_duplicates: bool,
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: run_json <input.json>");
    let input: Input = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let o = input.options;
    let opts = Options {
        keep: match o.keep.as_deref() {
            Some("longest") => KeepPolicy::Longest,
            Some("shortest") => KeepPolicy::Shortest,
            _ => KeepPolicy::First,
        },
        mode: if o.mode.as_deref() == Some("drop") { ClipMode::Drop } else { ClipMode::Crop },
        min_length: o.min_length,
        parallel_only: o.parallel_only,
        parallel_angle: o.parallel_angle.unwrap_or(30.0),
        segments: o.segments,
        keep_duplicates: o.keep_duplicates,
        ..Options::new(o.tolerance)
    };
    let geoms: Vec<Vec<Part>> = input
        .paths
        .iter()
        .map(|p| vec![Part::Line(LineString::from(p.iter().map(|c| (c[0], c[1])).collect::<Vec<_>>()))])
        .collect();

    let t = Instant::now();
    let r = deoverlap(&geoms, &opts);
    let seconds = t.elapsed().as_secs_f64();

    let kept: Vec<Vec<[f64; 2]>> = r
        .kept()
        .flatten()
        .filter_map(|p| match p {
            Part::Line(l) => Some(l.0.iter().map(|c| [c.x, c.y]).collect()),
            Part::Point(_) => None,
        })
        .collect();
    let out = serde_json::json!({
        "kept": kept,
        "removed_length": geom_length(&r.removed),
        "wholly_removed": r.wholly_removed.len(),
        "seconds": seconds,
    });
    println!("{out}");
}
