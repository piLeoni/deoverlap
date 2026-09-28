//! Same behaviour tests as the Python package (`tests/test_deoverlap.py`).

use deoverlap_core::{deoverlap, geom_length, ClipMode, Geometry, KeepPolicy, Options, Part};
use geo::{BooleanOps, Buffer, LineString, MultiLineString, Point};

fn line(pts: &[(f64, f64)]) -> Geometry {
    vec![Part::Line(LineString::from(pts.to_vec()))]
}

fn lines(g: &Geometry) -> Vec<&LineString<f64>> {
    g.iter()
        .filter_map(|p| match p {
            Part::Line(l) => Some(l),
            Part::Point(_) => None,
        })
        .collect()
}

fn kept_len(r: &deoverlap_core::DeoverlapResult, i: usize) -> f64 {
    geom_length(r.kept_parts[i].as_ref().expect("geometry kept"))
}

fn approx(a: f64, b: f64, abs: f64) -> bool {
    (a - b).abs() <= abs
}

#[test]
fn simple_line_overlap_crops_second() {
    let geoms = vec![line(&[(0., 0.), (2., 0.)]), line(&[(1., 0.05), (3., 0.05)])];
    let r = deoverlap(&geoms, &Options { keep_duplicates: true, ..Options::new(0.1) });
    assert_eq!(r.kept().count(), 2);
    assert_eq!(r.kept_parts[0].as_ref(), Some(&geoms[0]));
    assert!(kept_len(&r, 1) < geom_length(&geoms[1]));
    assert!(!r.removed.is_empty());
}

#[test]
fn segments_self_overlap_thins_a_narrow_ribbon() {
    let ring = line(&[(0., 0.), (10., 0.), (10., 0.08), (0., 0.08), (0., 0.)]);
    let total = geom_length(&ring);

    let plain = deoverlap(std::slice::from_ref(&ring), &Options { parallel_only: true, ..Options::new(0.1) });
    assert!(approx(kept_len(&plain, 0), total, total * 0.05));

    let selfed = deoverlap(
        &[ring],
        &Options { segments: true, parallel_only: true, min_length: 0.05, ..Options::new(0.1) },
    );
    assert!(kept_len(&selfed, 0) < total * 0.7);
}

#[test]
fn segment_adjacency_preserves_joints() {
    let elbow = line(&[(0., 0.), (2., 0.), (2., 2.)]);
    let r = deoverlap(std::slice::from_ref(&elbow), &Options { segments: true, ..Options::new(0.3) });
    assert!(approx(kept_len(&r, 0), geom_length(&elbow), 0.05));
}

#[test]
fn fully_engulfed_line_is_removed() {
    let geoms = vec![line(&[(0., 0.), (3., 0.)]), line(&[(1., 0.), (2., 0.)])];
    let r = deoverlap(&geoms, &Options { keep_duplicates: true, ..Options::new(0.2) });
    assert_eq!(r.kept().count(), 1);
    assert_eq!(r.wholly_removed, vec![1]);
    assert!(r.removed_parts[1].is_some());
}

#[test]
fn longest_keep_policy_prefers_longer_stroke() {
    let short = line(&[(0., 0.), (1., 0.)]);
    let long = line(&[(0.05, 0.), (3., 0.)]);
    let r = deoverlap(&[short, long.clone()], &Options { keep: KeepPolicy::Longest, ..Options::new(0.1) });
    assert!(approx(kept_len(&r, 1), geom_length(&long), geom_length(&long) * 0.05));
    assert!(r.wholly_removed.contains(&0) || r.kept_parts[0].is_none());
}

#[test]
fn group_keeps_split_ring_as_one_multipart() {
    let ring = line(&[(0., 0.), (2., 0.), (2., 2.), (0., 2.), (0., 0.)]);
    let cutter = line(&[(1., -1.), (1., 3.)]);
    let r = deoverlap(&[cutter, ring], &Options::new(0.15));
    assert!(lines(r.kept_parts[1].as_ref().unwrap()).len() >= 2);
}

#[test]
fn cut_ring_joins_across_its_start_point() {
    let ring = line(&[(0., 0.), (2., 0.), (2., 2.), (0., 2.), (0., 0.)]);
    let cutter = line(&[(1., -1.), (1., 3.)]);
    let r = deoverlap(&[cutter, ring], &Options::new(0.15));
    assert_eq!(lines(r.kept_parts[1].as_ref().unwrap()).len(), 2);
    let flat: usize = r.kept().map(|g| lines(g).len()).sum();
    assert_eq!(flat, 3);
}

#[test]
fn segments_removes_fold_back_between_neighbours() {
    let spike = line(&[(0., 0.), (6., 0.), (6.5, 0.4), (6., 0.08), (0., 0.08), (0., 0.)]);
    let r = deoverlap(
        &[spike],
        &Options {
            segments: true,
            parallel_only: true,
            min_length: 0.05,
            keep_duplicates: true,
            ..Options::new(0.1)
        },
    );
    let kept = MultiLineString(lines(r.kept_parts[0].as_ref().unwrap()).into_iter().cloned().collect());
    let back = LineString::from(vec![(6.5, 0.4), (6., 0.08)]).buffer(0.01);
    let overlap = back.clip(&kept, false);
    let overlap_len: f64 = overlap.0.iter().map(|l| geom_length(&[Part::Line(l.clone())])).sum();
    assert!(overlap_len < 0.1, "fold-back left {overlap_len}");
}

#[test]
fn segments_keeps_ordinary_corners() {
    let square = line(&[(0., 0.), (2., 0.), (2., 2.), (0., 2.), (0., 0.)]);
    let r = deoverlap(
        std::slice::from_ref(&square),
        &Options { segments: true, parallel_only: true, ..Options::new(0.3) },
    );
    assert!(approx(kept_len(&r, 0), geom_length(&square), 0.05));
}

#[test]
fn parallel_only_preserves_crossing() {
    let horizontal = line(&[(0., 0.), (4., 0.)]);
    let vertical = line(&[(2., -2.), (2., 2.)]);
    let geoms = [horizontal, vertical.clone()];
    let cropped = deoverlap(&geoms, &Options::new(0.3));
    let crossing = deoverlap(&geoms, &Options { parallel_only: true, parallel_angle: 30.0, ..Options::new(0.3) });
    assert!(kept_len(&crossing, 1) > kept_len(&cropped, 1));
    assert!(approx(kept_len(&crossing, 1), geom_length(&vertical), 0.05));
}

#[test]
fn parallel_only_uses_local_bearing() {
    let wall = line(&[(0., 0.), (0., 10.)]);
    let u_turn = line(&[(0.05, 9.), (0.05, 1.), (3., 1.), (3., 9.)]);
    let r = deoverlap(&[wall, u_turn.clone()], &Options { parallel_only: true, ..Options::new(0.1) });
    assert!(approx(kept_len(&r, 1), geom_length(&u_turn) - 8.0, 0.3));
    assert_eq!(lines(r.kept_parts[1].as_ref().unwrap()).len(), 1);
}

#[test]
fn min_length_drops_stubs() {
    let a = line(&[(0., 0.), (2., 0.)]);
    let b = line(&[(0.5, 0.02), (2.05, 0.02)]);
    let r = deoverlap(&[a, b], &Options { min_length: 0.1, keep_duplicates: true, ..Options::new(0.1) });
    match &r.kept_parts[1] {
        Some(g) => assert!(geom_length(g) >= 0.1),
        None => assert!(r.wholly_removed.contains(&1)),
    }
}

#[test]
fn drop_mode_discards_mostly_covered_line() {
    let a = line(&[(0., 0.), (3., 0.)]);
    let b = line(&[(0.5, 0.02), (3.5, 0.02)]);
    let geoms = [a, b.clone()];
    let cropped = deoverlap(&geoms, &Options { mode: ClipMode::Crop, ..Options::new(0.1) });
    let dropped = deoverlap(&geoms, &Options { mode: ClipMode::Drop, drop_fraction: 0.5, ..Options::new(0.1) });
    assert!(kept_len(&cropped, 1) < geom_length(&b));
    assert!(dropped.wholly_removed.contains(&1));
}

#[test]
fn mask_carries_across_stages() {
    let r1 = deoverlap(&[line(&[(0., 0.), (2., 0.)])], &Options::new(0.1));
    let batch2 = [line(&[(1., 0.05), (3., 0.05)])];
    let r2 = deoverlap(&batch2, &Options { mask: r1.mask, keep_duplicates: true, ..Options::new(0.1) });
    assert!(kept_len(&r2, 0) < geom_length(&batch2[0]));
    assert!(!r2.removed.is_empty());
}

fn wavy_lines() -> Vec<Geometry> {
    (0..12)
        .map(|k| {
            let a = k as f64 * 0.37;
            let pts: Vec<(f64, f64)> = (0..9)
                .map(|t| {
                    let t = t as f64;
                    (t * 0.731 + 0.05 * (t * 1.3 + a).sin(), 0.11 * k as f64 + 0.043 * (t + a).cos())
                })
                .collect();
            line(&pts)
        })
        .collect()
}

fn check_conservation(segments: bool) {
    let geoms = wavy_lines();
    let r = deoverlap(
        &geoms,
        &Options {
            keep: KeepPolicy::Longest,
            parallel_only: true,
            segments,
            keep_duplicates: true,
            ..Options::new(0.1)
        },
    );
    let total_in: f64 = geoms.iter().map(|g| geom_length(g)).sum();
    let kept: f64 = r.kept().map(|g| geom_length(g)).sum();
    let removed = geom_length(&r.removed);
    assert!(kept < total_in);
    assert!(approx(kept + removed, total_in, total_in * 1e-4), "kept {kept} + removed {removed} != {total_in}");
}

#[test]
fn kept_plus_removed_conserves_length() {
    check_conservation(false);
}

#[test]
fn kept_plus_removed_conserves_length_segments() {
    check_conservation(true);
}

#[test]
fn empty_input() {
    let r = deoverlap(&[], &Options::new(0.1));
    assert_eq!(r.kept().count(), 0);
    assert!(r.removed.is_empty());
    assert!(r.wholly_removed.is_empty());
}

#[test]
fn untouched_ring_is_returned_unchanged() {
    let circle: Vec<(f64, f64)> = (0..=32)
        .map(|i| {
            let t = i as f64 / 32.0 * std::f64::consts::TAU;
            (t.cos(), t.sin())
        })
        .collect();
    let ring = line(&circle);
    let far = line(&[(5., 0.), (6., 0.)]);
    let r = deoverlap(&[ring.clone(), far], &Options::new(0.1));
    assert_eq!(r.kept_parts[0].as_ref(), Some(&ring));
}

#[test]
fn points_are_clipped_by_corridors() {
    let base = line(&[(0., 0.), (2., 0.)]);
    let near = vec![Part::Point(Point::new(1.0, 0.05))];
    let far = vec![Part::Point(Point::new(1.0, 1.0))];
    let r = deoverlap(&[base, near, far], &Options::new(0.1));
    assert_eq!(r.wholly_removed, vec![1]);
    assert!(r.kept_parts[2].is_some());
}
