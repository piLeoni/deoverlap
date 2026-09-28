//! Python bindings for the Rust core, imported as `deoverlap._core`.
//!
//! Geometries cross the boundary as plain coordinates: a geometry is a list
//! of parts, a part is a list of `[x, y]`, and a single-coordinate part is a
//! point. The Shapely conversion lives on the Python side.

use deoverlap_core::{Mask, Options, Part, Polygon, Prefer};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type Coords = Vec<[f64; 2]>;
type PyGeometry = Vec<Coords>;
type PyPolygon = (Coords, Vec<Coords>);

type Output = (Vec<(usize, PyGeometry)>, Vec<Option<PyGeometry>>, Vec<Coords>, Vec<usize>);

fn to_part(coords: &Coords) -> Option<Part> {
    match coords.len() {
        0 => None,
        1 => Some(Part::Point(coords[0])),
        _ => Some(Part::Line(coords.clone())),
    }
}

fn from_part(part: &Part) -> Coords {
    match part {
        Part::Line(l) => l.clone(),
        Part::Point(p) => vec![*p],
    }
}

fn from_geometry(g: &[Part]) -> PyGeometry {
    g.iter().map(from_part).collect()
}

/// Returns `(kept, removed_parts, removed, wholly_removed)`: `kept` is
/// `(index, parts)` in the order geometries were kept, `removed_parts` is
/// aligned with the input. `mask` polygons clip every stroke. `progress`, if given, is called as
/// `progress(done, total)` about a hundred times per run.
#[pyfunction]
#[pyo3(signature = (
    geometries, tolerance, *, prefer = "longest", angle = 30.0, self_overlap = false,
    min_length = 0.0, drop = None, keep_duplicates = false, mask = Vec::new(), progress = None,
))]
#[allow(clippy::too_many_arguments)]
fn deoverlap(
    py: Python<'_>,
    geometries: Vec<PyGeometry>,
    tolerance: f64,
    prefer: &str,
    angle: f64,
    self_overlap: bool,
    min_length: f64,
    drop: Option<f64>,
    keep_duplicates: bool,
    mask: Vec<PyPolygon>,
    progress: Option<Py<PyAny>>,
) -> PyResult<Output> {
    let prefer = match prefer {
        "longest" => Prefer::Longest,
        "first" => Prefer::First,
        "shortest" => Prefer::Shortest,
        other => {
            return Err(PyValueError::new_err(format!(
                "prefer must be 'longest', 'first' or 'shortest', not {other:?}"
            )))
        }
    };
    if !(0.0..=90.0).contains(&angle) {
        return Err(PyValueError::new_err(format!("angle must be between 0 and 90 degrees, not {angle}")));
    }
    if let Some(f) = drop.filter(|f| !(0.0..=1.0).contains(f)) {
        return Err(PyValueError::new_err(format!("drop must be between 0 and 1, not {f}")));
    }
    let geoms: Vec<Vec<Part>> = geometries
        .iter()
        .map(|g| g.iter().filter_map(to_part).collect())
        .collect();
    let polygons = mask.into_iter().map(|(exterior, interiors)| Polygon { exterior, interiors }).collect();
    let mask = Mask { capsules: Vec::new(), polygons };
    let opts = Options { tolerance, prefer, angle, self_overlap, min_length, drop, keep_duplicates, mask };

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
    Ok((kept, removed_parts, removed, r.wholly_removed))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(deoverlap, m)?)?;
    Ok(())
}
