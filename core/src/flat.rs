//! A flat, language-neutral wire format for geometries.
//!
//! Bindings do not hand the engine nested vectors of `[x, y]`; they hand it
//! three flat buffers, which is what every host can produce cheaply:
//!
//! ```text
//! coords:  [x0, y0, x1, y1, ...]   every vertex, one flat run of f64
//! offsets: [0, 2, 5, 7, ...]       start of each part, in f64 units,
//!                                  with a final entry equal to coords.len()
//! kinds:   [0, 1, ...]             per part: 0 = line, 1 = point
//! ```
//!
//! A part with a single coordinate is a point; anything else is a line.
//! `kinds` is therefore redundant with `offsets`, but it lets a host state
//! its intent without the engine having to guess, and costs one byte per part.
//!
//! This is deliberately *not* WKT/WKB: those describe standalone geometry
//! objects, whereas the engine returns per-input bookkeeping (which piece
//! belongs to which input index) that no geometry format carries. A flat
//! buffer of numbers and indices is both smaller and exactly what Rust,
//! NumPy and `Float64Array` all read fastest.

use crate::{Coord, Geometry, Part};

/// Per-part kind tag in the flat format.
pub const KIND_LINE: u8 = 0;
/// Per-part kind tag in the flat format.
pub const KIND_POINT: u8 = 1;

/// A geometry as flat buffers: see the module docs for the layout.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FlatGeometry {
    pub coords: Vec<f64>,
    pub offsets: Vec<u32>,
    pub kinds: Vec<u8>,
}

impl FlatGeometry {
    /// True when there is nothing to process (no parts at all).
    pub fn is_empty(&self) -> bool {
        self.kinds.is_empty()
    }

    /// Number of parts.
    pub fn part_count(&self) -> usize {
        self.kinds.len()
    }

    /// The coordinates of part `i` as `[x, y]` pairs.
    pub fn part_coords(&self, i: usize) -> &[f64] {
        let lo = self.offsets[i] as usize;
        let hi = self.offsets[i + 1] as usize;
        &self.coords[lo..hi]
    }
}

/// Errors a malformed flat buffer can carry.
#[derive(Debug, PartialEq, Eq)]
pub enum FlatError {
    /// `coords.len()` is odd: a dangling x or y.
    OddCoords,
    /// `offsets` must start at 0 and end at `coords.len()`.
    BadOffsets,
    /// `offsets` must not go backwards.
    OffsetsNotSorted,
    /// `kinds.len()` must be `offsets.len() - 1`.
    KindCountMismatch,
    /// A part must hold at least one coordinate.
    EmptyPart,
    /// A part tagged `KIND_POINT` must hold exactly one coordinate.
    PointWithSeveralCoords,
    /// A part tagged `KIND_LINE` must hold at least two coordinates.
    LineWithOneCoord,
    /// Unknown kind tag.
    UnknownKind(u8),
}

impl core::fmt::Display for FlatError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            FlatError::OddCoords => write!(f, "coords must hold x, y pairs"),
            FlatError::BadOffsets => {
                write!(f, "offsets must start at 0 and end at the length of coords")
            }
            FlatError::OffsetsNotSorted => write!(f, "offsets must not go backwards"),
            FlatError::KindCountMismatch => {
                write!(f, "kinds must hold one entry per part (offsets.len() - 1)")
            }
            FlatError::EmptyPart => write!(f, "every part needs at least one coordinate"),
            FlatError::PointWithSeveralCoords => {
                write!(f, "a point part must hold exactly one coordinate")
            }
            FlatError::LineWithOneCoord => {
                write!(f, "a line part must hold at least two coordinates")
            }
            FlatError::UnknownKind(k) => write!(f, "unknown part kind {k}"),
        }
    }
}

impl std::error::Error for FlatError {}

impl FlatGeometry {
    /// Validate the buffers and turn them into the engine's `Geometry`.
    pub fn to_geometry(&self) -> Result<Geometry, FlatError> {
        if !self.coords.len().is_multiple_of(2) {
            return Err(FlatError::OddCoords);
        }
        if self.offsets.is_empty() {
            if self.coords.is_empty() && self.kinds.is_empty() {
                return Ok(Vec::new());
            }
            return Err(FlatError::BadOffsets);
        }
        if self.offsets[0] != 0 || *self.offsets.last().unwrap() as usize != self.coords.len() {
            return Err(FlatError::BadOffsets);
        }
        if self.kinds.len() + 1 != self.offsets.len() {
            return Err(FlatError::KindCountMismatch);
        }
        let mut parts = Vec::with_capacity(self.kinds.len());
        for i in 0..self.kinds.len() {
            let lo = self.offsets[i] as usize;
            let hi = self.offsets[i + 1] as usize;
            if hi < lo {
                return Err(FlatError::OffsetsNotSorted);
            }
            let raw = &self.coords[lo..hi];
            if !raw.len().is_multiple_of(2) {
                return Err(FlatError::OddCoords);
            }
            let pts: Vec<Coord> = raw.as_chunks::<2>().0.iter().map(|c| [c[0], c[1]]).collect();
            if pts.is_empty() {
                return Err(FlatError::EmptyPart);
            }
            parts.push(match self.kinds[i] {
                KIND_LINE => {
                    if pts.len() < 2 {
                        return Err(FlatError::LineWithOneCoord);
                    }
                    Part::Line(pts)
                }
                KIND_POINT => {
                    if pts.len() != 1 {
                        return Err(FlatError::PointWithSeveralCoords);
                    }
                    Part::Point(pts[0])
                }
                other => return Err(FlatError::UnknownKind(other)),
            });
        }
        Ok(parts)
    }

    /// Pack an engine `Geometry` into flat buffers.
    pub fn from_geometry(g: &Geometry) -> Self {
        let mut out = Self {
            coords: Vec::new(),
            // The leading 0 is the start of the first part; each part then
            // appends its end, so `offsets.len() == kinds.len() + 1`.
            offsets: vec![0],
            kinds: Vec::new(),
        };
        for part in g {
            let (pts, kind): (&[Coord], u8) = match part {
                Part::Line(l) => (l.as_slice(), KIND_LINE),
                Part::Point(p) => (core::slice::from_ref(p), KIND_POINT),
            };
            for c in pts {
                out.coords.push(c[0]);
                out.coords.push(c[1]);
            }
            out.kinds.push(kind);
            out.offsets.push(out.coords.len() as u32);
        }
        out
    }
}

/// A whole batch in the flat format: one [`FlatGeometry`] per input, and the
/// matching input index.
pub type FlatBatch = Vec<(usize, FlatGeometry)>;

/// The engine's output as flat buffers.
///
/// `kept` keeps the engine's processing order; `removed_parts` is aligned
/// with the input and only filled when `keep_duplicates` is set.
/// Engine mask for the next stage: corridor capsules plus any polygon clips.
#[derive(Clone, Debug, Default)]
pub struct FlatMask {
    pub capsules: Vec<crate::Capsule>,
    pub polygons: Vec<crate::Polygon>,
}

#[derive(Clone, Debug, Default)]
pub struct FlatResult {
    pub kept: FlatBatch,
    pub removed_parts: Vec<Option<FlatGeometry>>,
    pub removed: Vec<FlatGeometry>,
    pub wholly_removed: Vec<usize>,
    pub mask: FlatMask,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(a: (f64, f64), b: (f64, f64)) -> Part {
        Part::Line(vec![[a.0, a.1], [b.0, b.1]])
    }

    #[test]
    fn round_trips_a_batch() {
        let geom: Geometry = vec![line((0., 0.), (2., 0.)), Part::Point([1., 1.])];
        let flat = FlatGeometry::from_geometry(&geom);
        assert_eq!(flat.coords, vec![0., 0., 2., 0., 1., 1.]);
        assert_eq!(flat.offsets, vec![0, 4, 6]);
        assert_eq!(flat.kinds, vec![KIND_LINE, KIND_POINT]);
        assert_eq!(flat.to_geometry().unwrap(), geom);
    }

    #[test]
    fn empty_geometry_round_trips() {
        let flat = FlatGeometry::from_geometry(&Vec::new());
        assert!(flat.is_empty());
        assert_eq!(flat.offsets, vec![0]);
        assert_eq!(flat.to_geometry().unwrap(), Vec::new());
    }

    #[test]
    fn rejects_malformed_buffers() {
        let bad = |coords: Vec<f64>, offsets: Vec<u32>, kinds: Vec<u8>| {
            FlatGeometry { coords, offsets, kinds }.to_geometry()
        };
        assert_eq!(bad(vec![0., 0., 1.], vec![0, 3], vec![KIND_LINE]), Err(FlatError::OddCoords));
        assert_eq!(bad(vec![0., 0.], vec![2, 2], vec![KIND_LINE]), Err(FlatError::BadOffsets));
        assert_eq!(bad(vec![0., 0.], vec![0, 2], vec![]), Err(FlatError::KindCountMismatch));
        // A point part holding two coordinates.
        assert_eq!(
            bad(vec![0., 0., 1., 1.], vec![0, 4], vec![KIND_POINT]),
            Err(FlatError::PointWithSeveralCoords)
        );
        // A line part holding a single coordinate.
        assert_eq!(bad(vec![0., 0.], vec![0, 2], vec![KIND_LINE]), Err(FlatError::LineWithOneCoord));
        assert_eq!(bad(vec![0., 0.], vec![0, 2], vec![7]), Err(FlatError::UnknownKind(7)));
        // A part list that does not span the whole coord buffer; caught by the
        // offsets check, which runs before the kind count.
        assert_eq!(
            bad(vec![0., 0., 1., 1.], vec![0, 2], vec![KIND_LINE, KIND_LINE]),
            Err(FlatError::BadOffsets)
        );
        // Kind count that disagrees with the offsets, with a well-formed span.
        assert_eq!(
            bad(vec![0., 0., 1., 1.], vec![0, 2, 4], vec![KIND_LINE]),
            Err(FlatError::KindCountMismatch)
        );
    }

    #[test]
    fn part_coords_slices_each_part() {
        let geom: Geometry = vec![line((0., 0.), (2., 0.)), line((5., 5.), (6., 6.))];
        let flat = FlatGeometry::from_geometry(&geom);
        assert_eq!(flat.part_coords(0), &[0., 0., 2., 0.]);
        assert_eq!(flat.part_coords(1), &[5., 5., 6., 6.]);
        assert_eq!(flat.part_count(), 2);
        assert_eq!(flat.to_geometry().unwrap(), geom);
    }

    #[test]
    fn deoverlap_flat_matches_the_nested_engine() {
        use crate::{deoverlap, deoverlap_flat, Options, Prefer};

        let geoms: Vec<Geometry> = vec![
            vec![line((0., 0.), (2., 0.))],
            vec![line((1., 0.05), (3., 0.05))],
            vec![Part::Point([1., 0.02])],
        ];
        let opts = Options { prefer: Prefer::First, keep_duplicates: true, ..Options::new(0.1) };
        let nested = deoverlap(&geoms, &opts);
        let flat = deoverlap_flat(&geoms.iter().map(FlatGeometry::from_geometry).collect::<Vec<_>>(), &opts)
            .expect("valid buffers");

        let nested_kept: Vec<(usize, Geometry)> =
            nested.kept_order.iter().map(|&i| (i, nested.kept_parts[i].clone().unwrap())).collect();
        let flat_kept: Vec<(usize, Geometry)> = flat
            .kept
            .iter()
            .map(|(i, f)| (*i, f.to_geometry().unwrap()))
            .collect();
        assert_eq!(flat_kept, nested_kept);
        assert_eq!(flat.wholly_removed, nested.wholly_removed);
        assert_eq!(flat.removed_parts.len(), nested.removed_parts.len());
        for (a, b) in flat.removed_parts.iter().zip(&nested.removed_parts) {
            assert_eq!(
                a.as_ref().map(|f| f.to_geometry().unwrap()),
                b.clone(),
                "removed_parts disagree"
            );
        }
    }

    #[test]
    fn deoverlap_flat_propagates_bad_buffers() {
        use crate::{deoverlap_flat, Options};
        let broken = FlatGeometry { coords: vec![0., 0., 1.], offsets: vec![0, 3], kinds: vec![KIND_LINE] };
        assert_eq!(
            deoverlap_flat(&[broken], &Options::new(0.1)).unwrap_err(),
            FlatError::OddCoords
        );
    }
}
