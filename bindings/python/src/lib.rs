//! Python bindings for the Rust core, imported as `deoverlap._core`.
//!
//! Geometries cross the boundary as plain coordinates: a geometry is a list
//! of parts, a part is a list of `[x, y]`, and a single-coordinate part is a
//! point. The Shapely conversion lives on the Python side.
//!
//! `deoverlap_flat` speaks the language-neutral flat format documented in
//! `deoverlap_core::flat` instead: three flat buffers per geometry. It is the
//! entry point other bindings (Node) use, and the one this one is migrating
//! to.

use deoverlap_core::{
    deoverlap_flat as core_deoverlap_flat,
    deoverlap_flat_with_progress as core_deoverlap_flat_with_progress, Capsule, FlatError,
    FlatGeometry, Mask, Options, Part, Polygon, Prefer,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList};

type Coords = Vec<[f64; 2]>;
type PyGeometry = Vec<Coords>;
type PyPolygon = (Coords, Vec<Coords>);

type PyCapsule = ((f64, f64), (f64, f64), f64);
type PyMaskOut = (Vec<PyCapsule>, Vec<PyPolygon>);

type Output = (
    Vec<(usize, PyGeometry)>,
    Vec<Option<PyGeometry>>,
    Vec<Coords>,
    Vec<usize>,
    PyMaskOut,
);

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

/// Validate the shared keyword arguments into engine `Options`.
///
/// Both entry points take the same knobs, so the checks live in one place.
#[allow(clippy::too_many_arguments)]
fn check_options(
    tolerance: f64,
    prefer: &str,
    angle: f64,
    self_overlap: bool,
    min_length: f64,
    drop: Option<f64>,
    keep_duplicates: bool,
    mask: (Vec<PyPolygon>, Vec<PyCapsule>),
) -> PyResult<Options> {
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
    Ok(Options {
        tolerance,
        prefer,
        angle,
        self_overlap,
        min_length,
        drop,
        keep_duplicates,
        mask: build_mask(mask.0, mask.1),
    })
}

fn build_mask(polygons: Vec<PyPolygon>, capsules: Vec<PyCapsule>) -> Mask {
    Mask {
        capsules: capsules
            .into_iter()
            .map(|((ax, ay), (bx, by), radius)| Capsule {
                a: [ax, ay],
                b: [bx, by],
                radius,
            })
            .collect(),
        polygons: polygons
            .into_iter()
            .map(|(exterior, interiors)| Polygon { exterior, interiors })
            .collect(),
    }
}

fn export_mask(m: &Mask) -> PyMaskOut {
    let capsules = m
        .capsules
        .iter()
        .map(|c| ((c.a[0], c.a[1]), (c.b[0], c.b[1]), c.radius))
        .collect();
    let polygons = m
        .polygons
        .iter()
        .map(|p| (p.exterior.clone(), p.interiors.clone()))
        .collect();
    (capsules, polygons)
}

/// Returns `(kept, removed_parts, removed, wholly_removed)`: `kept` is
/// `(index, parts)` in the order geometries were kept, `removed_parts` is
/// aligned with the input. `mask` polygons clip every stroke. `progress`, if given, is called as
/// `progress(done, total)` about a hundred times per run.
#[pyfunction]
#[pyo3(signature = (
    geometries, tolerance, *, prefer = "longest", angle = 30.0, self_overlap = false,
    min_length = 0.0, drop = None, keep_duplicates = false, mask = Vec::new(),
    mask_capsules = Vec::new(), progress = None,
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
    mask_capsules: Vec<PyCapsule>,
    progress: Option<Py<PyAny>>,
) -> PyResult<Output> {
    let opts = check_options(
        tolerance,
        prefer,
        angle,
        self_overlap,
        min_length,
        drop,
        keep_duplicates,
        (mask, mask_capsules),
    )?;
    let geoms: Vec<Vec<Part>> = geometries
        .iter()
        .map(|g| g.iter().filter_map(to_part).collect())
        .collect();

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
    Ok((kept, removed_parts, removed, r.wholly_removed, export_mask(&r.mask)))
}

/// A geometry in the flat wire format, as a Python tuple.
///
/// `(coords, offsets, kinds)`: see `deoverlap_core::flat`. Accepts Python
/// lists/tuples/bytes; values are copied into Rust (same as the nested API).
#[pyclass(module = "deoverlap._core", frozen, skip_from_py_object)]
#[derive(Clone)]
struct FlatGeom {
    #[pyo3(get)]
    coords: Vec<f64>,
    #[pyo3(get)]
    offsets: Vec<u32>,
    #[pyo3(get)]
    kinds: Vec<u8>,
}

#[pymethods]
impl FlatGeom {
    #[new]
    fn new(coords: Vec<f64>, offsets: Vec<u32>, kinds: Vec<u8>) -> Self {
        Self { coords, offsets, kinds }
    }

    fn __repr__(&self) -> String {
        format!(
            "FlatGeom(coords={} f64, offsets={} parts, kinds={} parts)",
            self.coords.len(),
            self.offsets.len().saturating_sub(1),
            self.kinds.len()
        )
    }

    /// Flat buffers as plain Python lists, for hosts that do not have NumPy.
    fn buffers(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let t = (
            PyList::new(py, &self.coords)?,
            PyList::new(py, &self.offsets)?,
            PyList::new(py, &self.kinds)?,
        );
        Ok(t.into_pyobject(py)?.into_any().unbind())
    }
}

impl From<&FlatGeometry> for FlatGeom {
    fn from(f: &FlatGeometry) -> Self {
        Self { coords: f.coords.clone(), offsets: f.offsets.clone(), kinds: f.kinds.clone() }
    }
}

/// Reject a value whose type we cannot use, naming what we expected.
fn bad_type(name: &str, want: &str, got: &str) -> PyErr {
    PyTypeError::new_err(format!("{name} must be {want}; got {got}"))
}

fn type_name(obj: &Bound<'_, PyAny>) -> String {
    obj.get_type().name().map(|n| n.to_string()).unwrap_or_else(|_| "?".into())
}

/// Read a flat `[f64]` run from a Python sequence or a buffer object.
///
/// Lists and tuples take the `extract` path; NumPy arrays and `memoryview`
/// are read through the buffer protocol, which is what makes the flat format
/// worth having in the first place.
fn read_f64(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<f64>> {
    if let Ok(v) = obj.extract::<Vec<f64>>() {
        return Ok(v);
    }
    // `memoryview`/`array.array`: fall back to an element-wise read.
    if let Ok(seq) = obj.try_iter() {
        let mut out = Vec::new();
        for item in seq {
            out.push(item?.extract::<f64>().map_err(|_| {
                bad_type(name, "a sequence of float", &type_name(obj))
            })?);
        }
        return Ok(out);
    }
    Err(bad_type(name, "a sequence of float", &type_name(obj)))
}

/// Read a flat `[u32]` run from a Python sequence.
fn read_u32(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<u32>> {
    if let Ok(v) = obj.extract::<Vec<u32>>() {
        return Ok(v);
    }
    if let Ok(seq) = obj.try_iter() {
        let mut out = Vec::new();
        for item in seq {
            out.push(item?.extract::<u32>().map_err(|_| {
                bad_type(name, "a sequence of unsigned int", &type_name(obj))
            })?);
        }
        return Ok(out);
    }
    Err(bad_type(name, "a sequence of unsigned int", &type_name(obj)))
}

/// Read a flat `[u8]` run, accepting `bytes`/`bytearray` as well as sequences.
fn read_u8(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<u8>> {
    if let Ok(b) = obj.cast::<PyBytes>() {
        return Ok(b.as_bytes().to_vec());
    }
    if let Ok(v) = obj.extract::<Vec<u8>>() {
        return Ok(v);
    }
    if let Ok(seq) = obj.try_iter() {
        let mut out = Vec::new();
        for item in seq {
            out.push(item?.extract::<u8>().map_err(|_| {
                bad_type(name, "a bytes-like buffer", &type_name(obj))
            })?);
        }
        return Ok(out);
    }
    Err(bad_type(name, "a bytes-like buffer", &type_name(obj)))
}

/// Read one geometry as `(coords, offsets, kinds)`.
fn read_flat(obj: &Bound<'_, PyAny>) -> PyResult<FlatGeometry> {
    // Accept a `FlatGeom` returned by an earlier call, as well as a tuple.
    if let Ok(g) = obj.extract::<PyRef<FlatGeom>>() {
        return Ok(FlatGeometry {
            coords: g.coords.clone(),
            offsets: g.offsets.clone(),
            kinds: g.kinds.clone(),
        });
    }
    let (coords, offsets, kinds) = obj.extract::<(Bound<'_, PyAny>, Bound<'_, PyAny>, Bound<'_, PyAny>)>()
        .map_err(|_| bad_type("a geometry", "a (coords, offsets, kinds) tuple or FlatGeom", &type_name(obj)))?;
    Ok(FlatGeometry {
        coords: read_f64(&coords, "coords")?,
        offsets: read_u32(&offsets, "offsets")?,
        kinds: read_u8(&kinds, "kinds")?,
    })
}

fn flat_error(e: FlatError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

type FlatOutput = (Vec<(usize, FlatGeom)>, Vec<Option<FlatGeom>>, Vec<FlatGeom>, Vec<usize>);

/// [`deoverlap`] over the flat wire format.
///
/// Takes a sequence of `(coords, offsets, kinds)` and returns
/// `(kept, removed_parts, removed, wholly_removed)`, where every geometry is a
/// [`FlatGeom`] exposing `.coords`, `.offsets` and `.kinds` as Python lists.
/// The layout is the same for every binding; see `deoverlap_core::flat`.
#[pyfunction]
#[pyo3(signature = (
    geometries, tolerance, *, prefer = "longest", angle = 30.0, self_overlap = false,
    min_length = 0.0, drop = None, keep_duplicates = false, mask = Vec::new(), progress = None,
))]
#[allow(clippy::too_many_arguments)]
fn deoverlap_flat(
    py: Python<'_>,
    geometries: Vec<Bound<'_, PyAny>>,
    tolerance: f64,
    prefer: &str,
    angle: f64,
    self_overlap: bool,
    min_length: f64,
    drop: Option<f64>,
    keep_duplicates: bool,
    mask: Vec<PyPolygon>,
    progress: Option<Py<PyAny>>,
) -> PyResult<FlatOutput> {
    let opts = check_options(
        tolerance,
        prefer,
        angle,
        self_overlap,
        min_length,
        drop,
        keep_duplicates,
        (mask, Vec::new()),
    )?;
    let geoms: Vec<FlatGeometry> = geometries.iter().map(read_flat).collect::<PyResult<_>>()?;

    let mut callback_error: Option<PyErr> = None;
    let r = py.detach(|| {
        let Some(cb) = progress else {
            return core_deoverlap_flat(&geoms, &opts).map_err(flat_error);
        };
        let mut next = 0;
        core_deoverlap_flat_with_progress(&geoms, &opts, &mut |done, total| {
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
        .map_err(flat_error)
    });
    if let Some(e) = callback_error {
        return Err(e);
    }
    let r = r?;

    Ok((
        r.kept.iter().map(|(i, g)| (*i, FlatGeom::from(g))).collect(),
        r.removed_parts.iter().map(|g| g.as_ref().map(FlatGeom::from)).collect(),
        r.removed.iter().map(FlatGeom::from).collect(),
        r.wholly_removed,
    ))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FlatGeom>()?;
    m.add_function(wrap_pyfunction!(deoverlap, m)?)?;
    m.add_function(wrap_pyfunction!(deoverlap_flat, m)?)?;
    Ok(())
}
