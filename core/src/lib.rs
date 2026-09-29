//! Remove overlapping strokes from vector drawings.
//!
//! Geometries are processed in priority order; each kept stroke adds a
//! corridor of radius `tolerance` to a mask, and every later stroke loses
//! whatever falls inside a corridor running within `angle` degrees of it.
//!
//! Corridors are never built as polygons: the corridor of one straight edge
//! is a capsule, and the part of another edge inside it is computed exactly
//! (see `geom`).
//!
//! Polygons are not a separate input type: pass their rings as closed
//! lines, grouped in one [`Geometry`].

mod flat;
mod geom;
mod mask;
mod merge;

pub use flat::{FlatBatch, FlatError, FlatGeometry, FlatMask, FlatResult, KIND_LINE, KIND_POINT};

use std::f64::consts::PI;

use geom::{dist, lerp};
use mask::MaskIndex;
use merge::line_merge;

/// Clipped pieces are compared with a slack of `tolerance * SNAP_FRACTION`.
const SNAP_FRACTION: f64 = 1e-6;

/// In self-overlap mode, edges this many indices apart on the same path are
/// neighbours and never clip each other, unless they fold back.
const SEGMENT_ADJACENCY: i64 = 1;

/// Neighbouring edges fold back (a hairpin) when their headings differ by
/// more than 180° minus this.
const FOLD_BACK_TOL_DEG: f64 = 30.0;

/// Which geometry wins when two corridors collide.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Prefer {
    /// Longer geometries win.
    #[default]
    Longest,
    /// Earlier geometries win.
    First,
    /// Shorter geometries win.
    Shortest,
}

/// An `[x, y]` coordinate.
pub type Coord = [f64; 2];

/// One part of a geometry: a polyline (closed if first == last) or a point.
#[derive(Clone, Debug, PartialEq)]
pub enum Part {
    Line(Vec<Coord>),
    Point(Coord),
}

/// A geometry is a group of parts that are kept or removed together.
pub type Geometry = Vec<Part>;

/// Everything within `radius` of the segment `ab`.
#[derive(Clone, Debug, PartialEq)]
pub struct Capsule {
    pub a: Coord,
    pub b: Coord,
    pub radius: f64,
}

/// A polygon with holes; rings are closed (first == last).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Polygon {
    pub exterior: Vec<Coord>,
    pub interiors: Vec<Vec<Coord>>,
}

/// Areas that clip every stroke: the corridors of a previous run, or any
/// polygon to keep clear.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Mask {
    pub capsules: Vec<Capsule>,
    pub polygons: Vec<Polygon>,
}

#[derive(Clone, Debug)]
pub struct Options {
    /// Corridor radius: strokes closer than this to a kept stroke are cut.
    pub tolerance: f64,
    pub prefer: Prefer,
    /// Strokes overlap only where their local bearings differ by at most this
    /// many degrees; 90 or more cuts crossings too.
    pub angle: f64,
    /// Let a path overlap itself (e.g. the two sides of a thin outline).
    pub self_overlap: bool,
    /// Drop surviving pieces shorter than this.
    pub min_length: f64,
    /// Drop a whole stroke when more than this fraction of its length would
    /// be cut; `None` always crops.
    pub drop: Option<f64>,
    /// Collect the removed pieces in the result.
    pub keep_duplicates: bool,
    /// Corridors from a previous run that also clip this one.
    pub mask: Mask,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            tolerance: 0.0,
            prefer: Prefer::Longest,
            angle: 30.0,
            self_overlap: false,
            min_length: 0.0,
            drop: None,
            keep_duplicates: false,
            mask: Mask::default(),
        }
    }
}

impl Options {
    pub fn new(tolerance: f64) -> Self {
        Self { tolerance, ..Self::default() }
    }

    /// Whether bearings matter: below 90° some corridors are skipped.
    fn by_bearing(&self) -> bool {
        self.angle < 90.0
    }

    /// True if keeping `kept` out of `total` length counts as dropped.
    fn drops(&self, kept: f64, total: f64) -> bool {
        self.drop.is_some_and(|f| total > 0.0 && kept / total < 1.0 - f)
    }
}

#[derive(Clone, Debug, Default)]
pub struct DeoverlapResult {
    /// Surviving parts, aligned with the input (`None` = nothing kept).
    pub kept_parts: Vec<Option<Geometry>>,
    /// Indices of kept inputs in the order they were kept: priority order,
    /// or input order in self-overlap mode.
    pub kept_order: Vec<usize>,
    /// Removed pieces per input, filled when `keep_duplicates` is set.
    pub removed_parts: Vec<Option<Geometry>>,
    /// All removed pieces in processing order, when `keep_duplicates` is set.
    pub removed: Vec<Part>,
    /// Inputs removed entirely.
    pub wholly_removed: Vec<usize>,
    /// Corridors of everything kept, to pass as `mask` to a later run.
    pub mask: Mask,
}

impl DeoverlapResult {
    fn with_len(n: usize) -> Self {
        Self { kept_parts: vec![None; n], removed_parts: vec![None; n], ..Self::default() }
    }

    /// Kept geometries in the order they were kept.
    pub fn kept(&self) -> impl Iterator<Item = &Geometry> {
        self.kept_order.iter().filter_map(|&i| self.kept_parts[i].as_ref())
    }
}

pub fn deoverlap(geoms: &[Geometry], opts: &Options) -> DeoverlapResult {
    deoverlap_with_progress(geoms, opts, &mut |_, _| {})
}

/// [`deoverlap`] in the flat wire format, for bindings that receive and
/// return flat buffers instead of nested vectors.
///
/// This is the entry point Node and Python are moving to: it decodes once,
/// runs the same engine, and packs the result back into flat buffers.
pub fn deoverlap_flat(
    geoms: &[FlatGeometry],
    opts: &Options,
) -> Result<FlatResult, FlatError> {
    deoverlap_flat_with_progress(geoms, opts, &mut |_, _| {})
}

/// [`deoverlap_flat`], calling `progress(done, total)` as work advances.
pub fn deoverlap_flat_with_progress(
    geoms: &[FlatGeometry],
    opts: &Options,
    progress: &mut dyn FnMut(usize, usize),
) -> Result<FlatResult, FlatError> {
    let decoded: Vec<Geometry> = geoms
        .iter()
        .map(FlatGeometry::to_geometry)
        .collect::<Result<_, _>>()?;
    let r = deoverlap_with_progress(&decoded, opts, progress);
    Ok(FlatResult {
        kept: r
            .kept_order
            .iter()
            .filter_map(|&i| {
                r.kept_parts[i].as_ref().map(|g| (i, FlatGeometry::from_geometry(g)))
            })
            .collect(),
        removed_parts: r
            .removed_parts
            .iter()
            .map(|g| g.as_ref().map(FlatGeometry::from_geometry))
            .collect(),
        removed: r.removed.iter().map(|p| FlatGeometry::from_geometry(&vec![p.clone()])).collect(),
        wholly_removed: r.wholly_removed,
        mask: flat::FlatMask {
            capsules: r.mask.capsules,
            polygons: r.mask.polygons,
        },
    })
}

/// Like [`deoverlap`], calling `progress(done, total)` as work advances.
/// `total` counts geometries, or exploded edges in self-overlap mode.
pub fn deoverlap_with_progress(
    geoms: &[Geometry],
    opts: &Options,
    progress: &mut dyn FnMut(usize, usize),
) -> DeoverlapResult {
    let angle_tol_rad = opts.angle.to_radians();
    if opts.self_overlap {
        return deoverlap_self(geoms, opts, angle_tol_rad, progress);
    }

    let mut index = MaskIndex::new(&opts.mask, opts.by_bearing(), -1);
    let mut result = DeoverlapResult::with_len(geoms.len());
    let snap = opts.tolerance * SNAP_FRACTION;
    let want_removed = opts.keep_duplicates;

    let lengths: Vec<f64> = geoms.iter().map(|g| geom_length(g)).collect();
    for (done, i) in priority_order(&lengths, opts.prefer).into_iter().enumerate() {
        progress(done, geoms.len());
        let geom = &geoms[i];
        if geom.is_empty() {
            continue;
        }

        let mut kept_sub = Vec::new();
        let mut removed = Vec::new();
        for part in geom {
            let c = clip_part(part, &index, angle_tol_rad, None, snap, want_removed);
            kept_sub.extend(c.kept);
            removed.extend(c.removed);
        }

        let (reassembled, stubs) = filter_min_length(reassemble(kept_sub, snap), opts.min_length);
        removed.extend(stubs);
        if reassembled.is_empty() {
            result.remove_whole(i, geom, want_removed);
            continue;
        }

        if opts.drops(geom_length(&reassembled), lengths[i]) {
            result.remove_whole(i, geom, want_removed);
            continue;
        }

        if want_removed && !removed.is_empty() {
            let removed = reassemble(removed, snap);
            result.removed.extend(removed.iter().cloned());
            result.removed_parts[i] = Some(removed);
        }

        for part in &reassembled {
            add_corridors(&mut index, part, opts.tolerance, None);
        }
        result.kept_parts[i] = Some(reassembled);
        result.kept_order.push(i);
    }
    progress(geoms.len(), geoms.len());

    result.mask = index.into_mask();
    result
}

impl DeoverlapResult {
    fn remove_whole(&mut self, i: usize, geom: &Geometry, want_removed: bool) {
        self.wholly_removed.push(i);
        if want_removed {
            self.removed_parts[i] = Some(geom.clone());
            self.removed.extend(geom.iter().cloned());
        }
    }
}

/// Self-overlap path: work on exploded edges, then regroup by origin.
fn deoverlap_self(
    geoms: &[Geometry],
    opts: &Options,
    angle_tol_rad: f64,
    progress: &mut dyn FnMut(usize, usize),
) -> DeoverlapResult {
    let mut index = MaskIndex::new(&opts.mask, opts.by_bearing(), SEGMENT_ADJACENCY);
    let mut result = DeoverlapResult::with_len(geoms.len());
    let snap = opts.tolerance * SNAP_FRACTION;
    let want_removed = opts.keep_duplicates;

    let segs = explode_segments(geoms);
    if segs.is_empty() {
        result.mask = index.into_mask();
        return result;
    }

    let mut kept_by_origin: Vec<Vec<Part>> = vec![Vec::new(); geoms.len()];
    let mut removed_by_origin: Vec<Vec<Part>> = vec![Vec::new(); geoms.len()];
    let mut saw_origin = vec![false; geoms.len()];
    for s in &segs {
        saw_origin[s.origin] = true;
    }

    let parent_lengths: Vec<f64> = segs.iter().map(|s| s.parent_length).collect();
    for (done, si) in priority_order(&parent_lengths, opts.prefer).into_iter().enumerate() {
        progress(done, segs.len());
        let s = &segs[si];
        let whole = Part::Line(vec![s.a, s.b]);

        let c = clip_part(&whole, &index, angle_tol_rad, Some(&s.id), snap, want_removed);
        let dropped = c.kept.is_empty() || opts.drops(parts_length(&c.kept), dist(s.a, s.b));
        let (kept, stubs) = if dropped { (Vec::new(), Vec::new()) } else { filter_min_length(c.kept, opts.min_length) };
        if kept.is_empty() {
            if want_removed {
                result.removed.push(whole.clone());
                removed_by_origin[s.origin].push(whole);
            }
            continue;
        }

        if want_removed {
            let mut removed = c.removed;
            removed.extend(stubs);
            result.removed.extend(removed.iter().cloned());
            removed_by_origin[s.origin].extend(removed);
        }

        for part in &kept {
            add_corridors(&mut index, part, opts.tolerance, Some(s.id));
        }
        kept_by_origin[s.origin].extend(kept);
    }
    progress(segs.len(), segs.len());

    for (origin, pieces) in kept_by_origin.into_iter().enumerate() {
        if !saw_origin[origin] {
            continue;
        }
        let (reassembled, stubs) = filter_min_length(reassemble(pieces, snap), opts.min_length);
        if reassembled.is_empty() {
            result.wholly_removed.push(origin);
            if want_removed {
                result.removed_parts[origin] = Some(geoms[origin].clone());
            }
            continue;
        }
        if want_removed {
            let mut removed = std::mem::take(&mut removed_by_origin[origin]);
            removed.extend(stubs);
            if !removed.is_empty() {
                result.removed_parts[origin] = Some(reassemble(removed, snap));
            }
        }
        result.kept_parts[origin] = Some(reassembled);
        result.kept_order.push(origin);
    }

    result.mask = index.into_mask();
    result
}

/// One capsule per edge of a kept part (a disc for a point).
fn add_corridors(index: &mut MaskIndex, part: &Part, radius: f64, seg_id: Option<SegId>) {
    match part {
        Part::Line(l) => {
            let coords = dedupe(l);
            if coords.len() == 1 {
                index.add(Capsule { a: coords[0], b: coords[0], radius }, None, seg_id);
            }
            for w in coords.windows(2) {
                index.add(Capsule { a: w[0], b: w[1], radius }, Some(edge_angle(w[0], w[1])), seg_id);
            }
        }
        Part::Point(p) => index.add(Capsule { a: *p, b: *p, radius }, None, None),
    }
}

// ---------------------------------------------------------------------------
//  Segment ids
// ---------------------------------------------------------------------------

/// Position of an exploded edge on its chain (one input polyline).
#[derive(Clone, Copy, Debug)]
pub(crate) struct SegId {
    chain: usize,
    index: usize,
    count: usize,
    closed: bool,
    /// Direction of travel, radians.
    heading: f64,
}

impl SegId {
    /// True if `self` and `other` are neighbours on the same chain.
    fn adjacent(&self, other: &SegId, window: i64) -> bool {
        if self.chain != other.chain || window < 0 {
            return false;
        }
        let window = window as usize;
        let d = self.index.abs_diff(other.index);
        if d <= window {
            return true;
        }
        self.closed && self.count > 2 && self.count - d <= window
    }

    /// True if the chain doubles back: headings nearly opposite (a hairpin).
    fn folds_back(&self, other: &SegId) -> bool {
        let d = (self.heading - other.heading).abs() % (2.0 * PI);
        let d = d.min(2.0 * PI - d);
        d > PI - FOLD_BACK_TOL_DEG.to_radians()
    }
}

struct Segment {
    a: Coord,
    b: Coord,
    origin: usize,
    id: SegId,
    parent_length: f64,
}

fn explode_segments(geoms: &[Geometry]) -> Vec<Segment> {
    let mut out = Vec::new();
    let mut chain = 0;
    for (origin, geom) in geoms.iter().enumerate() {
        let parent_length = geom_length(geom);
        for part in geom {
            let Part::Line(ls) = part else { continue };
            let mut coords = dedupe(ls);
            if coords.len() < 2 {
                continue;
            }
            let closed = coords.len() > 2 && coords[0] == *coords.last().unwrap();
            if closed {
                coords.pop();
            }
            let count = if closed { coords.len() } else { coords.len() - 1 };
            for i in 0..count {
                let (a, b) = (coords[i], coords[(i + 1) % coords.len()]);
                if a == b {
                    continue;
                }
                let heading = (b[1] - a[1]).atan2(b[0] - a[0]);
                let id = SegId { chain, index: i, count, closed, heading };
                out.push(Segment { a, b, origin, id, parent_length });
            }
            chain += 1;
        }
    }
    out
}

// ---------------------------------------------------------------------------
//  Clipping
// ---------------------------------------------------------------------------

struct Clipped {
    kept: Vec<Part>,
    removed: Vec<Part>,
}

impl Clipped {
    fn untouched(part: &Part) -> Self {
        Self { kept: vec![part.clone()], removed: Vec::new() }
    }

    fn gone(part: &Part) -> Self {
        Self { kept: Vec::new(), removed: vec![part.clone()] }
    }
}

/// Clip a part edge by edge, each edge against the corridors within the
/// angle window of *that* edge.
///
/// One bearing per path (first to last vertex) would misjudge curves: a ramp
/// that runs alongside a road for a while can have a chord pointing elsewhere.
fn clip_part(
    part: &Part,
    mask: &MaskIndex,
    angle_tol_rad: f64,
    seg_id: Option<&SegId>,
    snap: f64,
    want_removed: bool,
) -> Clipped {
    let coords = match part {
        Part::Point(p) => {
            return if mask.covers_point(*p) { Clipped::gone(part) } else { Clipped::untouched(part) };
        }
        Part::Line(l) => dedupe(l),
    };
    if coords.len() < 2 {
        return match coords.first() {
            Some(&p) if mask.covers_point(p) => Clipped::gone(part),
            _ => Clipped::untouched(part),
        };
    }

    let mut kept = Runs::default();
    let mut removed = Runs::default();
    let mut changed = false;
    // Reused per edge: the corridor lookup and its filtered result.
    let mut ivs: Vec<(f64, f64)> = Vec::new();
    let mut covered: Vec<(f64, f64)> = Vec::new();
    for w in coords.windows(2) {
        let (a, b) = (w[0], w[1]);
        let slack = snap / dist(a, b);
        mask.covered_into(a, b, Some(edge_angle(a, b)), angle_tol_rad, seg_id, &mut ivs);
        covered.clear();
        covered.extend(
            ivs.iter()
                .copied()
                .filter(|(s0, s1)| s1 - s0 > slack)
                .map(|(s0, s1)| (if s0 < slack { 0.0 } else { s0 }, if s1 > 1.0 - slack { 1.0 } else { s1 })),
        );
        if covered.is_empty() {
            kept.extend(a, b, &[(0.0, 1.0)]);
            removed.extend(a, b, &[]);
            continue;
        }
        changed = true;
        kept.extend(a, b, &gaps(&covered, slack));
        if want_removed {
            removed.extend(a, b, &covered);
        }
    }
    if !changed {
        return Clipped::untouched(part);
    }
    let (start, end) = (coords[0], *coords.last().unwrap());
    Clipped { kept: kept.finish(start, end), removed: removed.finish(start, end) }
}

/// The parts of `[0, 1]` not covered by sorted, disjoint `covered`, ignoring
/// slivers of `slack` or less.
fn gaps(covered: &[(f64, f64)], slack: f64) -> Vec<(f64, f64)> {
    let mut out = Vec::new();
    let mut t = 0.0;
    for &(s0, s1) in covered {
        if s0 - t > slack {
            out.push((t, s0));
        }
        t = s1;
    }
    if 1.0 - t > slack {
        out.push((t, 1.0));
    }
    out
}

/// Polylines built from per-edge intervals: a run continues across a vertex
/// when one edge's interval ends at 1 and the next edge's starts at 0.
#[derive(Default)]
struct Runs {
    done: Vec<Vec<Coord>>,
    open: Option<Vec<Coord>>,
}

impl Runs {
    fn extend(&mut self, a: Coord, b: Coord, intervals: &[(f64, f64)]) {
        for &(s0, s1) in intervals {
            let p1 = if s1 >= 1.0 { b } else { lerp(a, b, s1) };
            match self.open.as_mut() {
                Some(run) if s0 <= 0.0 => run.push(p1),
                _ => {
                    self.close();
                    let p0 = if s0 <= 0.0 { a } else { lerp(a, b, s0) };
                    self.open = Some(vec![p0, p1]);
                }
            }
            if s1 < 1.0 {
                self.close();
            }
        }
        if intervals.last().is_none_or(|iv| iv.1 < 1.0) {
            self.close();
        }
    }

    fn close(&mut self) {
        if let Some(run) = self.open.take() {
            self.done.push(run);
        }
    }

    /// Finished runs; on a closed ring, the runs through its start vertex
    /// are joined into one.
    fn finish(mut self, start: Coord, end: Coord) -> Vec<Part> {
        self.close();
        let mut runs = self.done;
        if start == end && runs.len() > 1 && runs[0][0] == start && *runs.last().unwrap().last().unwrap() == end {
            let first = runs.remove(0);
            runs.last_mut().unwrap().extend(first.into_iter().skip(1));
        }
        runs.into_iter().map(Part::Line).collect()
    }
}

// ---------------------------------------------------------------------------
//  Reassembly and helpers
// ---------------------------------------------------------------------------

/// Rejoin pieces that meet end to end: consecutive edges in self-overlap
/// mode, or parts of one geometry that touch.
fn reassemble(parts: Vec<Part>, snap: f64) -> Vec<Part> {
    let mut lines = Vec::new();
    let mut points = Vec::new();
    for p in parts {
        match p {
            Part::Line(l) => lines.push(l),
            Part::Point(pt) => points.push(Part::Point(pt)),
        }
    }
    let mut out: Vec<Part> = line_merge(lines, snap).into_iter().map(Part::Line).collect();
    out.extend(points);
    out
}

/// Split `parts` into those at least `min_length` long (points always pass)
/// and the short stubs.
fn filter_min_length(parts: Vec<Part>, min_length: f64) -> (Vec<Part>, Vec<Part>) {
    if min_length <= 0.0 {
        return (parts, Vec::new());
    }
    parts.into_iter().partition(|p| match p {
        Part::Line(l) => line_length(l) >= min_length,
        Part::Point(_) => true,
    })
}

fn priority_order(scores: &[f64], prefer: Prefer) -> Vec<usize> {
    let mut idxs: Vec<usize> = (0..scores.len()).collect();
    match prefer {
        Prefer::First => {}
        Prefer::Longest => idxs.sort_by(|&a, &b| scores[b].total_cmp(&scores[a])),
        Prefer::Shortest => idxs.sort_by(|&a, &b| scores[a].total_cmp(&scores[b])),
    }
    idxs
}

fn dedupe(coords: &[Coord]) -> Vec<Coord> {
    let mut out: Vec<Coord> = Vec::with_capacity(coords.len());
    for &c in coords {
        if out.last() != Some(&c) {
            out.push(c);
        }
    }
    out
}

/// Undirected bearing of an edge, in [0, π).
fn edge_angle(a: Coord, b: Coord) -> f64 {
    (b[1] - a[1]).atan2(b[0] - a[0]).rem_euclid(PI)
}

pub(crate) fn angle_diff(a: f64, b: f64) -> f64 {
    let d = (a - b).abs() % PI;
    d.min(PI - d)
}

fn line_length(ls: &[Coord]) -> f64 {
    ls.windows(2).map(|w| dist(w[0], w[1])).sum()
}

fn part_length(p: &Part) -> f64 {
    match p {
        Part::Line(l) => line_length(l),
        Part::Point(_) => 0.0,
    }
}

fn parts_length(parts: &[Part]) -> f64 {
    parts.iter().map(part_length).sum()
}

/// Total length of a geometry's lines (a polygon's perimeter).
pub fn geom_length(geom: &[Part]) -> f64 {
    parts_length(geom)
}
