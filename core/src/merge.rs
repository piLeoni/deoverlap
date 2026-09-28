use std::collections::HashMap;

use crate::Coord;

type Line = Vec<Coord>;

/// Join lines that meet end to end, like GEOS `LineMerge`.
///
/// Endpoints closer than `eps` count as the same node, because clipped pieces
/// drift off the original vertices by float noise. Lines are only joined
/// through nodes where exactly two line ends meet.
pub(crate) fn line_merge(lines: Vec<Line>, eps: f64) -> Vec<Line> {
    let lines: Vec<Line> = lines.into_iter().filter(|l| l.len() >= 2).collect();
    if lines.len() < 2 {
        return lines;
    }

    let mut nodes = NodeIndex::new(eps);
    let ends: Vec<(usize, usize)> = lines
        .iter()
        .map(|l| (nodes.node_of(l[0]), nodes.node_of(*l.last().unwrap())))
        .collect();

    let mut incident: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];
    for (i, &(s, e)) in ends.iter().enumerate() {
        incident[s].push(i);
        incident[e].push(i);
    }

    let mut used = vec![false; lines.len()];
    let mut out = Vec::new();
    for start in 0..lines.len() {
        if used[start] {
            continue;
        }
        used[start] = true;
        let mut coords: Line = lines[start].clone();
        let (mut head, mut tail) = ends[start];

        // Walk forward from the tail, then backward from the head.
        for forward in [true, false] {
            loop {
                let node = if forward { tail } else { head };
                if incident[node].len() != 2 {
                    break;
                }
                let Some(&next) = incident[node].iter().find(|&&j| !used[j]) else {
                    break;
                };
                used[next] = true;
                let (s, e) = ends[next];
                let mut piece = lines[next].clone();
                let far = if s == node { e } else { piece.reverse(); s };
                if forward {
                    coords.extend(piece.into_iter().skip(1));
                    tail = far;
                } else {
                    piece.reverse();
                    piece.pop();
                    piece.extend(coords);
                    coords = piece;
                    head = far;
                }
            }
        }
        out.push(coords);
    }
    out
}

struct NodeIndex {
    eps: f64,
    coords: Vec<Coord>,
    grid: HashMap<(i64, i64), Vec<usize>>,
}

impl NodeIndex {
    fn new(eps: f64) -> Self {
        Self { eps: eps.max(f64::MIN_POSITIVE), coords: Vec::new(), grid: HashMap::new() }
    }

    fn len(&self) -> usize {
        self.coords.len()
    }

    fn cell(&self, c: Coord) -> (i64, i64) {
        ((c[0] / self.eps).floor() as i64, (c[1] / self.eps).floor() as i64)
    }

    fn node_of(&mut self, c: Coord) -> usize {
        let (cx, cy) = self.cell(c);
        for dx in -1..=1 {
            for dy in -1..=1 {
                if let Some(ids) = self.grid.get(&(cx + dx, cy + dy)) {
                    for &id in ids {
                        let o = self.coords[id];
                        if crate::geom::dist(o, c) <= self.eps {
                            return id;
                        }
                    }
                }
            }
        }
        let id = self.coords.len();
        self.coords.push(c);
        self.grid.entry((cx, cy)).or_default().push(id);
        id
    }
}
