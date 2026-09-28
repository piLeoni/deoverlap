//! Plain 2D geometry on `[x, y]` coordinates.
//!
//! The corridor of a segment is a capsule: every point within `radius` of it.
//! A capsule is convex, so the part of a straight edge inside it is a single
//! interval of the edge's parameter, found in closed form.

use crate::{Coord, Polygon};

pub(crate) fn sub(a: Coord, b: Coord) -> Coord {
    [a[0] - b[0], a[1] - b[1]]
}

pub(crate) fn dot(a: Coord, b: Coord) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

pub(crate) fn cross(a: Coord, b: Coord) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}

pub(crate) fn dist(a: Coord, b: Coord) -> f64 {
    (b[0] - a[0]).hypot(b[1] - a[1])
}

pub(crate) fn lerp(a: Coord, b: Coord, t: f64) -> Coord {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]
}

/// Parameter range `[s0, s1]` of `p + s (q - p)`, clamped to `[0, 1]`, that
/// lies within `radius` of segment `ab`.
pub(crate) fn capsule_interval(
    p: Coord,
    q: Coord,
    a: Coord,
    b: Coord,
    radius: f64,
) -> Option<(f64, f64)> {
    let d = sub(q, p);
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    let mut take = |iv: Option<(f64, f64)>| {
        if let Some((s0, s1)) = iv {
            lo = lo.min(s0);
            hi = hi.max(s1);
        }
    };
    take(disc_interval(p, d, a, radius));
    let e = sub(b, a);
    let len = dot(e, e).sqrt();
    if len > 0.0 {
        take(disc_interval(p, d, b, radius));
        take(slab_interval(
            p,
            d,
            a,
            [e[0] / len, e[1] / len],
            len,
            radius,
        ));
    }
    let (lo, hi) = (lo.max(0.0), hi.min(1.0));
    (lo < hi).then_some((lo, hi))
}

/// `|p + s d - c| <= r`, solved for `s`.
fn disc_interval(p: Coord, d: Coord, c: Coord, r: f64) -> Option<(f64, f64)> {
    let f = sub(p, c);
    let a = dot(d, d);
    let b = dot(f, d);
    let disc = b * b - a * (dot(f, f) - r * r);
    if a == 0.0 || disc < 0.0 {
        return None;
    }
    let root = disc.sqrt();
    Some(((-b - root) / a, (-b + root) / a))
}

/// The rectangle between the capsule's end caps: along `e` within
/// `[0, len]`, across it within `[-r, r]`.
fn slab_interval(p: Coord, d: Coord, a: Coord, e: Coord, len: f64, r: f64) -> Option<(f64, f64)> {
    let f = sub(p, a);
    let mut s0 = f64::NEG_INFINITY;
    let mut s1 = f64::INFINITY;
    for (start, rate, lo, hi) in [
        (dot(f, e), dot(d, e), 0.0, len),
        (cross(f, e), cross(d, e), -r, r),
    ] {
        if rate == 0.0 {
            if start < lo || start > hi {
                return None;
            }
            continue;
        }
        let (t0, t1) = ((lo - start) / rate, (hi - start) / rate);
        s0 = s0.max(t0.min(t1));
        s1 = s1.min(t0.max(t1));
    }
    (s0 <= s1).then_some((s0, s1))
}

pub(crate) fn point_in_capsule(p: Coord, a: Coord, b: Coord, radius: f64) -> bool {
    let e = sub(b, a);
    let len2 = dot(e, e);
    let t = if len2 > 0.0 {
        (dot(sub(p, a), e) / len2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let c = lerp(a, b, t);
    let f = sub(p, c);
    dot(f, f) <= radius * radius
}

/// Even-odd point in polygon: inside the exterior and outside every hole.
pub(crate) fn point_in_polygon(p: Coord, poly: &Polygon) -> bool {
    let mut inside = false;
    for ring in std::iter::once(&poly.exterior).chain(&poly.interiors) {
        let n = ring.len();
        if n < 3 {
            continue;
        }
        let mut j = n - 1;
        for i in 0..n {
            let (a, b) = (ring[i], ring[j]);
            if (a[1] > p[1]) != (b[1] > p[1])
                && p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0]
            {
                inside = !inside;
            }
            j = i;
        }
    }
    inside
}

/// Parameter ranges of `p + s (q - p)` inside `poly`: split the edge where
/// it crosses a ring, then test the middle of each piece.
pub(crate) fn polygon_intervals(p: Coord, q: Coord, poly: &Polygon, out: &mut Vec<(f64, f64)>) {
    let d = sub(q, p);
    let mut cuts = vec![0.0, 1.0];
    for ring in std::iter::once(&poly.exterior).chain(&poly.interiors) {
        for w in ring.windows(2) {
            let e = sub(w[1], w[0]);
            let denom = cross(d, e);
            if denom == 0.0 {
                continue;
            }
            let f = sub(w[0], p);
            let s = cross(f, e) / denom;
            let t = cross(f, d) / denom;
            if (0.0..=1.0).contains(&t) && s > 0.0 && s < 1.0 {
                cuts.push(s);
            }
        }
    }
    cuts.sort_by(f64::total_cmp);
    for w in cuts.windows(2) {
        if w[1] > w[0] && point_in_polygon(lerp(p, q, (w[0] + w[1]) / 2.0), poly) {
            out.push((w[0], w[1]));
        }
    }
}

/// Sort and merge overlapping intervals.
pub(crate) fn merge_intervals(mut ivs: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
    ivs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut out: Vec<(f64, f64)> = Vec::with_capacity(ivs.len());
    for (s0, s1) in ivs {
        match out.last_mut() {
            Some(last) if s0 <= last.1 => last.1 = last.1.max(s1),
            _ => out.push((s0, s1)),
        }
    }
    out
}
