//! Python bindings for the Rust core, imported as `deoverlap._core`.
//!
//! Geometries cross the boundary as plain coordinates: a geometry is a list
//! of parts, a part is a list of `[x, y]`, and a single-coordinate part is a
//! point. The Shapely conversion lives on the Python side.

use deoverlap_core::{ClipMode, KeepPolicy, Options, Part};
use geo::{LineString, Point, Polygon};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type Coords = Vec<[f64; 2]>;
type PyGeometry = Vec<Coords>;
type PyPolygon = (Coords, Vec<Coords>);

type Output = (
    Vec<(usize, PyGeometry)>,
    Vec<Option<PyGeometry>>,
    Vec<Coords>,
    Vec<usize>,
    Vec<PyPolygon>,
);

fn to_ring(coords: &Coords) -> LineString<f64> {
    LineString::from(coords.iter().map(|c| (c[0], c[1])).collect::<Vec<_>>())
}

fn to_part(coords: &Coords) -> Option<Part> {
    match coords.len() {
        0 => None,
        1 => Some(Part::Point(Point::new(coords[0][0], coords[0][1]))),
        _ => Some(Part::Line(to_ring(coords))),
    }
}

fn from_part(part: &Part) -> Coords {
    match part {
        Part::Line(l) => l.0.iter().map(|c| [c.x, c.y]).collect(),
        Part::Point(p) => vec![[p.x(), p.y()]],
    }
}

fn from_geometry(g: &[Part]) -> PyGeometry {
    g.iter().map(from_part).collect()
}

fn from_ring(ring: &LineString<f64>) -> Coords {
    ring.0.iter().map(|c| [c.x, c.y]).collect()
}

/// Returns `(kept, removed_parts, removed, wholly_removed, mask)`: `kept` is
/// `(index, parts)` in the order geometries were kept, `removed_parts` is
/// aligned with the input. `progress`, if given, is called as
/// `progress(done, total)` about a hundred times per run.
#[pyfunction]
#[pyo3(signature = (
    geometries, tolerance, *, keep = "first", mode = "crop", min_length = 0.0,
    drop_fraction = 0.5, parallel_only = false, parallel_angle = 30.0,
    segments = false, segment_adjacency = 1, keep_duplicates = false, mask = Vec::new(),
    progress = None,
))]
#[allow(clippy::too_many_arguments)]
fn deoverlap(
    py: Python<'_>,
    geometries: Vec<PyGeometry>,
    tolerance: f64,
    keep: &str,
    mode: &str,
    min_length: f64,
    drop_fraction: f64,
    parallel_only: bool,
    parallel_angle: f64,
    segments: bool,
    segment_adjacency: i64,
    keep_duplicates: bool,
    mask: Vec<PyPolygon>,
    progress: Option<Py<PyAny>>,
) -> PyResult<Output> {
    let keep = match keep {
        "first" => KeepPolicy::First,
        "longest" => KeepPolicy::Longest,
        "shortest" => KeepPolicy::Shortest,
        other => return Err(PyValueError::new_err(format!("unknown keep policy: {other:?}"))),
    };
    let mode = match mode {
        "crop" => ClipMode::Crop,
        "drop" => ClipMode::Drop,
        other => return Err(PyValueError::new_err(format!("unknown clip mode: {other:?}"))),
    };
    let geoms: Vec<Vec<Part>> = geometries
        .iter()
        .map(|g| g.iter().filter_map(to_part).collect())
        .collect();
    let mask = mask
        .iter()
        .map(|(ext, ints)| Polygon::new(to_ring(ext), ints.iter().map(to_ring).collect()))
        .collect();
    let opts = Options {
        tolerance,
        keep,
        mode,
        min_length,
        drop_fraction,
        parallel_only,
        parallel_angle,
        segments,
        segment_adjacency,
        keep_duplicates,
        mask,
    };

    let mut callback_error: Option<PyErr> = None;
    let r = py.detach(|| {
        let Some(cb) = progress else {
            return deoverlap_core::deoverlap(&geoms, &opts);
        };
        let mut next = 0;
        deoverlap_core::deoverlap_with_progress(&geoms, &opts, &mut |done, total| {
            if (done < next && done < total) || callback_error.is_some() {
                return;
            }
            next = done + (total / 100).max(1);
            Python::attach(|py| {
                if let Err(e) = cb.call1(py, (done, total)) {
                    callback_error = Some(e);
                }
            });
        })
    });
    if let Some(e) = callback_error {
        return Err(e);
    }

    let kept = r
        .kept_order
        .iter()
        .filter_map(|&i| r.kept_parts[i].as_deref().map(|g| (i, from_geometry(g))))
        .collect();
    let removed_parts = r.removed_parts.iter().map(|g| g.as_deref().map(from_geometry)).collect();
    let removed = r.removed.iter().map(from_part).collect();
    let mask = r
        .mask
        .iter()
        .map(|p| (from_ring(p.exterior()), p.interiors().iter().map(from_ring).collect()))
        .collect();
    Ok((kept, removed_parts, removed, r.wholly_removed, mask))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(deoverlap, m)?)?;
    Ok(())
}
