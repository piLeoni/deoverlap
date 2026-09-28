//! Remove overlapping strokes from vector drawings.
//!
//! Built on the `geo` crate. Geometries are processed in priority order; each
//! kept stroke adds a corridor of radius `tolerance` to a mask, and every
//! later stroke loses whatever falls inside a corridor running within
//! `angle` degrees of it.
//!
//! Polygons are not a separate input type: pass their rings as closed
//! lines, grouped in one [`Geometry`].

mod mask;
mod merge;

use std::f64::consts::PI;

use geo::bool_ops::FillRule;
use geo::{BooleanOps, BoundingRect, Buffer, Coord, Intersects, LineString, MultiLineString, Point, Polygon, Rect};

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

/// One part of a geometry: a polyline (closed if first == last) or a point.
#[derive(Clone, Debug, PartialEq)]
pub enum Part {
    Line(LineString<f64>),
    Point(Point<f64>),
}

/// A geometry is a group of parts that are kept or removed together.
pub type Geometry = Vec<Part>;

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
    pub mask: Vec<Polygon<f64>>,
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
            mask: Vec::new(),
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
    pub mask: Vec<Polygon<f64>>,
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

    let by_bearing = opts.by_bearing();
    let mut index = MaskIndex::new(&opts.mask, by_bearing, -1);
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
            let c = if by_bearing {
                clip_local(part, &index, angle_tol_rad, snap, want_removed)
            } else {
                clip_one(part, &index, None, angle_tol_rad, None, want_removed)
            };
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
            match part {
                Part::Line(ls) if by_bearing => {
                    for (a, b) in edges(ls) {
                        let edge = LineString::new(vec![a, b]);
                        index.add(edge.buffer(opts.tolerance), Some(edge_angle(a, b)), None);
                    }
                }
                Part::Line(ls) => index.add(ls.buffer(opts.tolerance), None, None),
                Part::Point(p) => index.add(p.buffer(opts.tolerance), None, None),
            }
        }
        result.kept_parts[i] = Some(reassembled);
        result.kept_order.push(i);
    }
    progress(geoms.len(), geoms.len());

    result.mask = index.into_polygons();
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
    let by_bearing = opts.by_bearing();
    let mut index = MaskIndex::new(&opts.mask, by_bearing, SEGMENT_ADJACENCY);
    let mut result = DeoverlapResult::with_len(geoms.len());
    let snap = opts.tolerance * SNAP_FRACTION;
    let want_removed = opts.keep_duplicates;

    let segs = explode_segments(geoms);
    if segs.is_empty() {
        result.mask = index.into_polygons();
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
        let whole = Part::Line(LineString::new(vec![s.a, s.b]));
        let angle = by_bearing.then(|| edge_angle(s.a, s.b));

        let c = clip_one(&whole, &index, angle, angle_tol_rad, Some(&s.id), want_removed);
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
            if let Part::Line(ls) = part {
                index.add(ls.buffer(opts.tolerance), angle, Some(s.id));
            }
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

    result.mask = index.into_polygons();
    result
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
    a: Coord<f64>,
    b: Coord<f64>,
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
            let mut coords = dedupe(&ls.0);
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
                let heading = (b.y - a.y).atan2(b.x - a.x);
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
}

fn clip_one(
    part: &Part,
    mask: &MaskIndex,
    angle: Option<f64>,
    angle_tol_rad: f64,
    seg_id: Option<&SegId>,
    want_removed: bool,
) -> Clipped {
    let Some(rect) = part_rect(part) else {
        return Clipped::untouched(part);
    };
    let Some(local) = mask.local_mask(rect, angle, angle_tol_rad, seg_id) else {
        return Clipped::untouched(part);
    };
    match part {
        Part::Point(p) => {
            if local.intersects(p) {
                Clipped { kept: Vec::new(), removed: vec![part.clone()] }
            } else {
                Clipped::untouched(part)
            }
        }
        Part::Line(ls) => {
            if !local.intersects(ls) {
                return Clipped::untouched(part);
            }
            let subject = MultiLineString(vec![ls.clone()]);
            let pieces = |invert| -> Vec<Part> {
                local
                    .clip_with_fill_rule(&subject, invert, FillRule::NonZero)
                    .0
                    .into_iter()
                    .filter(|l| line_length(l) > 0.0)
                    .map(Part::Line)
                    .collect()
            };
            let kept = pieces(true);
            let removed = if want_removed { pieces(false) } else { Vec::new() };
            Clipped { kept, removed }
        }
    }
}

/// Clip edge by edge, each against corridors parallel to *that* edge.
///
/// One bearing per path (first to last vertex) misjudges curves: a ramp that
/// runs alongside a road for a while can have a chord pointing elsewhere.
fn clip_local(part: &Part, mask: &MaskIndex, angle_tol_rad: f64, snap: f64, want_removed: bool) -> Clipped {
    let near = part_rect(part).is_some_and(|r| mask.near(r));
    let Part::Line(ls) = part else {
        return clip_one(part, mask, None, angle_tol_rad, None, want_removed);
    };
    if !near {
        return clip_one(part, mask, None, angle_tol_rad, None, want_removed);
    }
    let mut kept = Vec::new();
    let mut removed = Vec::new();
    let mut changed = false;
    for (a, b) in edges(ls) {
        let edge = Part::Line(LineString::new(vec![a, b]));
        let c = clip_one(&edge, mask, Some(edge_angle(a, b)), angle_tol_rad, None, want_removed);
        if parts_length(&c.kept) < dist(a, b) - snap {
            changed = true;
        }
        kept.extend(c.kept);
        removed.extend(c.removed);
    }
    if !changed {
        return Clipped::untouched(part);
    }
    Clipped { kept: reassemble(kept, snap), removed: reassemble(removed, snap) }
}

// ---------------------------------------------------------------------------
//  Reassembly and helpers
// ---------------------------------------------------------------------------

/// Rejoin pieces that meet end to end: the arc through a ring's start vertex,
/// or consecutive edges in self-overlap mode.
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

fn dedupe(coords: &[Coord<f64>]) -> Vec<Coord<f64>> {
    let mut out: Vec<Coord<f64>> = Vec::with_capacity(coords.len());
    for &c in coords {
        if out.last() != Some(&c) {
            out.push(c);
        }
    }
    out
}

/// 2-point edges of a polyline, in order, skipping repeated vertices.
fn edges(ls: &LineString<f64>) -> impl Iterator<Item = (Coord<f64>, Coord<f64>)> {
    let coords = dedupe(&ls.0);
    (1..coords.len()).map(move |i| (coords[i - 1], coords[i]))
}

/// Undirected bearing of an edge, in [0, π).
fn edge_angle(a: Coord<f64>, b: Coord<f64>) -> f64 {
    (b.y - a.y).atan2(b.x - a.x).rem_euclid(PI)
}

pub(crate) fn angle_diff(a: f64, b: f64) -> f64 {
    let d = (a - b).abs() % PI;
    d.min(PI - d)
}

fn dist(a: Coord<f64>, b: Coord<f64>) -> f64 {
    (b.x - a.x).hypot(b.y - a.y)
}

fn line_length(ls: &LineString<f64>) -> f64 {
    ls.0.windows(2).map(|w| dist(w[0], w[1])).sum()
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

fn part_rect(p: &Part) -> Option<Rect<f64>> {
    match p {
        Part::Line(l) => l.bounding_rect(),
        Part::Point(pt) => Some(Rect::new(pt.0, pt.0)),
    }
}
