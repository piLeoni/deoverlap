use rstar::primitives::{GeomWithData, Rectangle};
use rstar::{RTree, AABB};

use crate::geom::{
    capsule_interval, merge_intervals, point_in_capsule, point_in_polygon, polygon_intervals,
};
use crate::{Capsule, Coord, Mask, Polygon, SegId};

type Entry = GeomWithData<Rectangle<[f64; 2]>, usize>;

/// Corridors of everything kept so far, with R-trees over their bounding
/// boxes.
///
/// Each capsule keeps the bearing and segment id it was drawn with, which
/// `covered` filters on. Polygons (from an earlier run) always apply.
pub(crate) struct MaskIndex {
    capsules: Vec<Capsule>,
    angles: Vec<Option<f64>>,
    seg_ids: Vec<Option<SegId>>,
    tree: RTree<Entry>,
    polygons: Vec<Polygon>,
    polygon_tree: RTree<Entry>,
    by_bearing: bool,
    segment_adjacency: i64,
}

fn bounds(points: impl IntoIterator<Item = Coord>, pad: f64) -> Option<AABB<[f64; 2]>> {
    let mut it = points.into_iter();
    let first = it.next()?;
    let (mut lo, mut hi) = (first, first);
    for p in it {
        lo = [lo[0].min(p[0]), lo[1].min(p[1])];
        hi = [hi[0].max(p[0]), hi[1].max(p[1])];
    }
    Some(AABB::from_corners(
        [lo[0] - pad, lo[1] - pad],
        [hi[0] + pad, hi[1] + pad],
    ))
}

fn entry(b: AABB<[f64; 2]>, id: usize) -> Entry {
    GeomWithData::new(Rectangle::from_corners(b.lower(), b.upper()), id)
}

impl MaskIndex {
    pub(crate) fn new(initial: &Mask, by_bearing: bool, segment_adjacency: i64) -> Self {
        let mut index = Self {
            capsules: Vec::new(),
            angles: Vec::new(),
            seg_ids: Vec::new(),
            tree: RTree::new(),
            polygons: Vec::new(),
            polygon_tree: RTree::new(),
            by_bearing,
            segment_adjacency: segment_adjacency.max(0),
        };
        for c in &initial.capsules {
            index.push(c.clone(), None, None);
        }
        let polygon_entries = initial
            .polygons
            .iter()
            .enumerate()
            .filter_map(|(i, p)| bounds(p.exterior.iter().copied(), 0.0).map(|b| entry(b, i)))
            .collect();
        index.polygon_tree = RTree::bulk_load(polygon_entries);
        index.polygons = initial.polygons.clone();
        index
    }

    pub(crate) fn add(&mut self, capsule: Capsule, angle: Option<f64>, seg_id: Option<SegId>) {
        let angle = if self.by_bearing { angle } else { None };
        self.push(capsule, angle, seg_id);
    }

    fn push(&mut self, capsule: Capsule, angle: Option<f64>, seg_id: Option<SegId>) {
        let b = bounds([capsule.a, capsule.b], capsule.radius).expect("two points");
        let id = self.capsules.len();
        self.capsules.push(capsule);
        self.angles.push(angle);
        self.seg_ids.push(seg_id);
        self.tree.insert(entry(b, id));
    }

    /// Merged parameter ranges of edge `pq` that fall inside a corridor
    /// running within `angle_tol_rad` of `angle`.
    pub(crate) fn covered(
        &self,
        p: Coord,
        q: Coord,
        angle: Option<f64>,
        angle_tol_rad: f64,
        seg_id: Option<&SegId>,
    ) -> Vec<(f64, f64)> {
        let b = bounds([p, q], 0.0).expect("two points");
        let mut ivs = Vec::new();
        for e in self.tree.locate_in_envelope_intersecting(&b) {
            let i = e.data;
            if let (Some(a), Some(s)) = (seg_id, self.seg_ids[i].as_ref()) {
                if a.adjacent(s, self.segment_adjacency) && !a.folds_back(s) {
                    continue;
                }
            }
            if self.by_bearing {
                if let (Some(a), Some(k)) = (angle, self.angles[i]) {
                    if crate::angle_diff(a, k) > angle_tol_rad {
                        continue;
                    }
                }
            }
            let c = &self.capsules[i];
            if let Some(iv) = capsule_interval(p, q, c.a, c.b, c.radius) {
                ivs.push(iv);
            }
        }
        for e in self.polygon_tree.locate_in_envelope_intersecting(&b) {
            polygon_intervals(p, q, &self.polygons[e.data], &mut ivs);
        }
        merge_intervals(ivs)
    }

    pub(crate) fn covers_point(&self, p: Coord) -> bool {
        let b = AABB::from_point(p);
        self.tree.locate_in_envelope_intersecting(&b).any(|e| {
            let c = &self.capsules[e.data];
            point_in_capsule(p, c.a, c.b, c.radius)
        }) || self
            .polygon_tree
            .locate_in_envelope_intersecting(&b)
            .any(|e| point_in_polygon(p, &self.polygons[e.data]))
    }

    pub(crate) fn into_mask(self) -> Mask {
        Mask {
            capsules: self.capsules,
            polygons: self.polygons,
        }
    }
}
