use geo::{BoundingRect, MultiPolygon, Orient, Polygon, Rect};
use rstar::primitives::{GeomWithData, Rectangle};
use rstar::{RTree, AABB};

use crate::SegId;

type Entry = GeomWithData<Rectangle<[f64; 2]>, usize>;

/// Corridor polygons ("sausages") of everything kept so far, with an R-tree
/// over their bounding boxes.
///
/// Corridors are never dissolved into one another: each keeps the bearing
/// and segment id it was drawn with, which `local_mask` filters on.
pub(crate) struct MaskIndex {
    polys: Vec<MultiPolygon<f64>>,
    angles: Vec<Option<f64>>,
    seg_ids: Vec<Option<SegId>>,
    tree: RTree<Entry>,
    parallel_only: bool,
    segment_adjacency: i64,
}

fn aabb(r: Rect<f64>) -> AABB<[f64; 2]> {
    AABB::from_corners([r.min().x, r.min().y], [r.max().x, r.max().y])
}

impl MaskIndex {
    pub(crate) fn new(initial: &[Polygon<f64>], parallel_only: bool, segment_adjacency: i64) -> Self {
        let mut index = Self {
            polys: Vec::new(),
            angles: Vec::new(),
            seg_ids: Vec::new(),
            tree: RTree::new(),
            parallel_only,
            segment_adjacency: segment_adjacency.max(0),
        };
        for p in initial {
            index.push(MultiPolygon(vec![p.clone()]), None, None);
        }
        index
    }

    pub(crate) fn add(&mut self, corridor: MultiPolygon<f64>, angle: Option<f64>, seg_id: Option<SegId>) {
        let angle = if self.parallel_only { angle } else { None };
        self.push(corridor, angle, seg_id);
    }

    fn push(&mut self, corridor: MultiPolygon<f64>, angle: Option<f64>, seg_id: Option<SegId>) {
        let Some(rect) = corridor.bounding_rect() else {
            return;
        };
        let id = self.polys.len();
        // Consistent winding lets `clip` treat overlapping corridors as a
        // union under the non-zero fill rule.
        self.polys.push(corridor.orient(geo::orient::Direction::Default));
        self.angles.push(angle);
        self.seg_ids.push(seg_id);
        let b = aabb(rect);
        self.tree.insert(GeomWithData::new(Rectangle::from_corners(b.lower(), b.upper()), id));
    }

    pub(crate) fn near(&self, rect: Rect<f64>) -> bool {
        self.tree.locate_in_envelope_intersecting(&aabb(rect)).next().is_some()
    }

    /// Corridors that may clip a geometry with bounding box `rect`, all
    /// rings concatenated into one multipolygon (to be read as non-zero).
    pub(crate) fn local_mask(
        &self,
        rect: Rect<f64>,
        angle: Option<f64>,
        angle_tol_rad: f64,
        seg_id: Option<&SegId>,
    ) -> Option<MultiPolygon<f64>> {
        let mut ids: Vec<usize> = self
            .tree
            .locate_in_envelope_intersecting(&aabb(rect))
            .map(|e| e.data)
            .filter(|&i| {
                if let (Some(a), Some(b)) = (seg_id, self.seg_ids[i].as_ref()) {
                    if a.adjacent(b, self.segment_adjacency) && !a.folds_back(b, angle_tol_rad) {
                        return false;
                    }
                }
                if self.parallel_only {
                    if let (Some(a), Some(b)) = (angle, self.angles[i]) {
                        if crate::angle_diff(a, b) > angle_tol_rad {
                            return false;
                        }
                    }
                }
                true
            })
            .collect();
        if ids.is_empty() {
            return None;
        }
        ids.sort_unstable();
        let polys = ids.iter().flat_map(|&i| self.polys[i].0.iter().cloned()).collect();
        Some(MultiPolygon(polys))
    }

    pub(crate) fn into_polygons(self) -> Vec<Polygon<f64>> {
        self.polys.into_iter().flat_map(|m| m.0).collect()
    }
}
