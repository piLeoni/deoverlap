//! Node.js bindings for `deoverlap-core` (flat wire format only).
//!
//! The public Python API stays on Shapely; this crate is the native path for
//! JavaScript hosts. It does not affect Python wheel performance.

use deoverlap_core::{deoverlap_flat, Capsule, FlatGeometry, Mask, Options, Polygon, Prefer};
use napi::bindgen_prelude::*;
use napi_derive::napi;

#[napi(object)]
pub struct FlatGeometryJs {
    pub coords: Vec<f64>,
    pub offsets: Vec<u32>,
    pub kinds: Vec<u8>,
}

#[napi(object)]
pub struct CapsuleJs {
    pub ax: f64,
    pub ay: f64,
    pub bx: f64,
    pub by: f64,
    pub radius: f64,
}

#[napi(object)]
pub struct PolygonMaskJs {
    pub exterior: Vec<f64>,
    pub interiors: Vec<Vec<f64>>,
}

#[napi(object)]
pub struct MaskJs {
    pub capsules: Vec<CapsuleJs>,
    pub polygons: Vec<PolygonMaskJs>,
}

#[napi(object)]
pub struct DeoverlapOptionsJs {
    pub prefer: Option<String>,
    pub angle: Option<f64>,
    pub self_overlap: Option<bool>,
    pub min_length: Option<f64>,
    pub drop: Option<f64>,
    pub keep_duplicates: Option<bool>,
    pub mask: Option<MaskJs>,
}

#[napi(object)]
pub struct KeptGeometryJs {
    pub index: u32,
    pub geometry: FlatGeometryJs,
}

#[napi(object)]
pub struct RemovedPartJs {
    pub index: u32,
    pub geometry: Option<FlatGeometryJs>,
}

#[napi(object)]
pub struct DeoverlapResultJs {
    pub kept: Vec<KeptGeometryJs>,
    pub wholly_removed: Vec<u32>,
    pub mask: MaskJs,
    pub removed: Vec<FlatGeometryJs>,
    pub removed_parts: Vec<RemovedPartJs>,
}

fn to_flat(g: &FlatGeometryJs) -> Result<FlatGeometry> {
    let f = FlatGeometry {
        coords: g.coords.clone(),
        offsets: g.offsets.clone(),
        kinds: g.kinds.clone(),
    };
    f.to_geometry().map_err(|e| Error::from_reason(e.to_string()))?;
    Ok(f)
}

fn from_flat(f: &FlatGeometry) -> FlatGeometryJs {
    FlatGeometryJs {
        coords: f.coords.clone(),
        offsets: f.offsets.clone(),
        kinds: f.kinds.clone(),
    }
}

fn parse_mask(m: &MaskJs) -> Result<Mask> {
    let capsules = m
        .capsules
        .iter()
        .map(|c| Capsule {
            a: [c.ax, c.ay],
            b: [c.bx, c.by],
            radius: c.radius,
        })
        .collect();
    let mut polygons = Vec::with_capacity(m.polygons.len());
    for p in &m.polygons {
        if p.exterior.len() % 2 != 0 {
            return Err(Error::from_reason("mask polygon exterior must be x,y pairs"));
        }
        let exterior: Vec<[f64; 2]> = p
            .exterior
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| [c[0], c[1]])
            .collect();
        let mut interiors = Vec::with_capacity(p.interiors.len());
        for ring in &p.interiors {
            if ring.len() % 2 != 0 {
                return Err(Error::from_reason("mask polygon interior must be x,y pairs"));
            }
            interiors.push(
                ring.as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| [c[0], c[1]])
                    .collect(),
            );
        }
        polygons.push(Polygon { exterior, interiors });
    }
    Ok(Mask { capsules, polygons })
}

fn export_mask(m: &Mask) -> MaskJs {
    MaskJs {
        capsules: m
            .capsules
            .iter()
            .map(|c| CapsuleJs {
                ax: c.a[0],
                ay: c.a[1],
                bx: c.b[0],
                by: c.b[1],
                radius: c.radius,
            })
            .collect(),
        polygons: m
            .polygons
            .iter()
            .map(|p| PolygonMaskJs {
                exterior: p.exterior.iter().flat_map(|c| [c[0], c[1]]).collect(),
                interiors: p
                    .interiors
                    .iter()
                    .map(|ring| ring.iter().flat_map(|c| [c[0], c[1]]).collect())
                    .collect(),
            })
            .collect(),
    }
}

fn parse_options(tolerance: f64, opts: Option<DeoverlapOptionsJs>) -> Result<Options> {
    let o = opts.unwrap_or(DeoverlapOptionsJs {
        prefer: None,
        angle: None,
        self_overlap: None,
        min_length: None,
        drop: None,
        keep_duplicates: None,
        mask: None,
    });
    let prefer = match o.prefer.as_deref().unwrap_or("longest") {
        "longest" => Prefer::Longest,
        "first" => Prefer::First,
        "shortest" => Prefer::Shortest,
        other => return Err(Error::from_reason(format!("prefer must be longest|first|shortest, not {other}"))),
    };
    let angle = o.angle.unwrap_or(30.0);
    if !(0.0..=90.0).contains(&angle) {
        return Err(Error::from_reason(format!("angle must be 0..=90, not {angle}")));
    }
    if let Some(f) = o.drop.filter(|f| !(0.0..=1.0).contains(f)) {
        return Err(Error::from_reason(format!("drop must be 0..=1, not {f}")));
    }
    let mask = match o.mask {
        Some(m) => parse_mask(&m)?,
        None => Mask::default(),
    };
    Ok(Options {
        tolerance,
        prefer,
        angle,
        self_overlap: o.self_overlap.unwrap_or(false),
        min_length: o.min_length.unwrap_or(0.0),
        drop: o.drop,
        keep_duplicates: o.keep_duplicates.unwrap_or(false),
        mask,
    })
}

/// De-overlap strokes in the flat `(coords, offsets, kinds)` format.
#[napi(js_name = "deoverlap")]
pub fn deoverlap_js(
    geometries: Vec<FlatGeometryJs>,
    tolerance: f64,
    options: Option<DeoverlapOptionsJs>,
) -> Result<DeoverlapResultJs> {
    let geoms: Vec<FlatGeometry> = geometries.iter().map(to_flat).collect::<Result<_>>()?;
    let opts = parse_options(tolerance, options)?;
    let r = deoverlap_flat(&geoms, &opts).map_err(|e| Error::from_reason(e.to_string()))?;
    Ok(DeoverlapResultJs {
        kept: r
            .kept
            .iter()
            .map(|(i, g)| KeptGeometryJs {
                index: *i as u32,
                geometry: from_flat(g),
            })
            .collect(),
        wholly_removed: r.wholly_removed.iter().map(|&i| i as u32).collect(),
        mask: export_mask(&Mask {
            capsules: r.mask.capsules,
            polygons: r.mask.polygons,
        }),
        removed: r.removed.iter().map(from_flat).collect(),
        removed_parts: r
            .removed_parts
            .iter()
            .enumerate()
            .map(|(i, g)| RemovedPartJs {
                index: i as u32,
                geometry: g.as_ref().map(from_flat),
            })
            .collect(),
    })
}
