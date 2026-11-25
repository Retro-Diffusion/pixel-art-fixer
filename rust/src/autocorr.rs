//! Grid detection via banded autocorrelation + cepstrum. Mirrors
//! detector/autocorr.py (axis_estimate, local_count, detect).
//!
//! Feature maps are pre-oriented: every map is row-major (lines x extent)
//! with the scan axis horizontal, so the y-axis maps arrive transposed.

use crate::acf::*;
use crate::gray::*;
use rayon::prelude::*;
use rustfft::FftPlanner;

/// One pre-oriented feature map with its fusion weight.
pub struct FeatMap {
    pub data: Vec<f32>,
    pub lines: usize,
    pub extent: usize,
    pub weight: f64,
}

pub struct AxisEstimate {
    pub step: f64,
    pub cands: Vec<(f64, f64)>,
    pub acf: Vec<f64>,
}

fn arange(start: f64, stop: f64, step: f64) -> Vec<f64> {
    let n = ((stop - start) / step).ceil().max(0.0) as usize;
    (0..n).map(|i| start + i as f64 * step).collect()
}

pub fn axis_estimate(
    maps: &[FeatMap],
    extent: usize,
    planner: &mut FftPlanner<f64>,
) -> AxisEstimate {
    let steps = arange(MIN_STEP, MAX_STEP.min(extent as f64 / 4.0), STEP_GRID);
    let ns = steps.len();
    let mut raw = vec![0f64; ns];
    let mut ac_sum = vec![0f64; extent];

    // evaluate every (map, band) task in parallel; accumulate in the
    // reference order afterwards so float sums stay bit-identical
    let mut tasks: Vec<(usize, usize, f64)> = Vec::new();
    for (mi, m) in maps.iter().enumerate() {
        for (band, bw) in [(12usize, 0.5f64), (BAND, 1.0), (m.lines, 1.0)] {
            tasks.push((mi, band, bw));
        }
    }
    let results: Vec<(Vec<f64>, Vec<f64>)> = tasks
        .par_iter()
        .map(|&(mi, band, _)| {
            let m = &maps[mi];
            let mut local = FftPlanner::new();
            let prof = band_profiles(&m.data, m.lines, m.extent, band);
            let ac = band_acf(&prof, &mut local);
            let combs: Vec<f64> = steps
                .par_iter()
                .map(|&s| comb_score(&ac, s, 24, 12.0))
                .collect();
            (ac, combs)
        })
        .collect();
    for (t, &(mi, _, bw)) in tasks.iter().enumerate() {
        let wgt = maps[mi].weight;
        let (ac, combs) = &results[t];
        for i in 0..extent {
            ac_sum[i] += wgt * bw * ac[i];
        }
        for i in 0..ns {
            raw[i] += wgt * bw * combs[i];
        }
    }
    drop(results);

    // cepstrum vote on the primary feature map
    let prof0 = band_profiles(&maps[0].data, maps[0].lines, maps[0].extent, BAND);
    let c = band_cepstrum(&prof0, planner);
    let raw_max = raw.iter().cloned().fold(f64::MIN, f64::max).max(1e-9);
    for (i, &s) in steps.iter().enumerate() {
        let cz = ceps_score(&c, s).max(0.0);
        raw[i] += 0.1 * cz * raw[i].max(0.0) / raw_max;
    }

    // subharmonic suppression (selection only)
    let mut pen = vec![0f64; ns];
    for m in [2f64, 3.0, 4.0, 5.0] {
        for i in 0..ns {
            let s_sub = steps[i] / m;
            if s_sub < steps[0] {
                continue;
            }
            let pos = ((s_sub - steps[0]) / STEP_GRID).max(0.0);
            let val = interp_at(&raw, pos).max(0.0);
            if val > pen[i] {
                pen[i] = val;
            }
        }
    }
    let sel: Vec<f64> = (0..ns).map(|i| raw[i] - 0.5 * pen[i]).collect();

    // local maxima of sel; parabolic refine on raw
    let mut loc: Vec<usize> = Vec::new();
    for i in 1..ns.saturating_sub(1) {
        if sel[i] > sel[i - 1] && sel[i] >= sel[i + 1] {
            loc.push(i);
        }
    }
    loc.sort_by(|&a, &b| sel[b].partial_cmp(&sel[a]).unwrap());
    loc.truncate(8);
    let mut cands: Vec<(f64, f64)> = Vec::new();
    for &i in &loc {
        let s = if i > 0 && i < ns - 1 {
            let d = (raw[i - 1] - raw[i + 1])
                / (2.0 * (raw[i - 1] - 2.0 * raw[i] + raw[i + 1]) + 1e-12);
            steps[i] + d.clamp(-1.0, 1.0) * STEP_GRID
        } else {
            steps[i]
        };
        cands.push((s, sel[i]));
    }
    if cands.is_empty() {
        let mut i = 0;
        for j in 0..ns {
            if sel[j] > sel[i] {
                i = j;
            }
        }
        cands.push((steps[i], sel[i]));
    }

    // near-tie disambiguation via peak-train quality on the pooled ACF
    if cands.len() >= 2 && cands[1].1 >= 0.85 * cands[0].1 {
        let a0 = ac_sum[0].max(1e-12);
        let ac_norm: Vec<f64> = ac_sum.iter().map(|&v| v / a0).collect();
        let top: Vec<(f64, f64)> = cands
            .iter()
            .cloned()
            .filter(|c| c.1 >= 0.85 * cands[0].1)
            .take(3)
            .collect();
        let mut best_j = 0;
        let mut best_v = f64::MIN;
        for (j, &(s, z)) in top.iter().enumerate() {
            let q = train_quality(&ac_norm, s, 20);
            let v = z * (0.1 + q);
            if v > best_v {
                best_v = v;
                best_j = j;
            }
        }
        if best_j != 0 {
            let chosen = top[best_j];
            let mut rest: Vec<(f64, f64)> =
                cands.iter().cloned().filter(|&c| c != chosen).collect();
            let mut new_cands = vec![chosen];
            new_cands.append(&mut rest);
            cands = new_cands;
        }
    }

    // precision: fit the ACF peak train around each top candidate
    for c in cands.iter_mut() {
        c.0 = refine_step_acf(&ac_sum, c.0, 24);
    }
    AxisEstimate {
        step: cands[0].0,
        cands,
        acf: ac_sum,
    }
}

/// Drift-aware cell count via windowed local ACF step integration.
pub fn local_count(maps: &[FeatMap], extent: usize, s0: f64, planner: &mut FftPlanner<f64>) -> f64 {
    let uniform = extent as f64 / s0;
    let nw = (extent as f64 / (14.0 * s0)).clamp(1.0, 8.0) as usize;
    if nw < 2 || uniform < 96.0 {
        return uniform;
    }
    // np.linspace(0, extent, nw+1).astype(int)
    let edges: Vec<usize> = (0..=nw)
        .map(|i| (extent as f64 * i as f64 / nw as f64) as usize)
        .collect();
    let _ = planner;
    let counts: Vec<f64> = (0..nw)
        .into_par_iter()
        .map(|wdw| {
            let (a, b) = (edges[wdw], edges[wdw + 1]);
            let wlen = b - a;
            let mut local = FftPlanner::new();
            let mut ac_loc = vec![0f64; wlen];
            for m in maps {
                // column slice [a, b) of the pre-oriented map
                let mut sl = vec![0f32; m.lines * wlen];
                for y in 0..m.lines {
                    let src = y * m.extent + a;
                    sl[y * wlen..(y + 1) * wlen].copy_from_slice(&m.data[src..src + wlen]);
                }
                for (band, bw) in [(BAND, 1.0f64), (m.lines, 1.0)] {
                    let prof = band_profiles(&sl, m.lines, wlen, band);
                    let ac = band_acf(&prof, &mut local);
                    for i in 0..wlen {
                        ac_loc[i] += m.weight * bw * ac[i];
                    }
                }
            }
            let scan: Vec<f64> = arange(0.85, 1.18, 0.01).iter().map(|f| s0 * f).collect();
            let sc: Vec<f64> = scan
                .iter()
                .map(|&s| comb_score(&ac_loc, s, 12, 8.0))
                .collect();
            let s_glob = comb_score(&ac_loc, s0, 12, 8.0);
            let sc_max = sc.iter().cloned().fold(f64::MIN, f64::max);
            let s_i = if sc_max > (1.6 * s_glob.max(0.0)).max(0.02) {
                let mut i = 0;
                for j in 0..sc.len() {
                    if sc[j] > sc[i] {
                        i = j;
                    }
                }
                refine_step_acf(&ac_loc, scan[i], 12)
            } else {
                refine_step_acf(&ac_loc, s0, 10)
            };
            let s_i = s_i.clamp(0.82 * s0, 1.2 * s0);
            wlen as f64 / s_i
        })
        .collect();
    let mut total = 0f64;
    for c in counts {
        total += c;
    }
    if (total - uniform).abs() >= 2.5 {
        total
    } else {
        uniform
    }
}

pub struct Detection {
    pub step_x: f64,
    pub step_y: f64,
    pub cols: usize,
    pub rows: usize,
    pub candidates: Vec<(f64, f64)>,
}

/// Build the pre-oriented feature maps for one axis.
/// For axis=x the maps are used as-is; for axis=y they are transposed.
pub fn build_maps(g: &[f32], gq: &[f32], w: usize, h: usize, y_axis: bool) -> Vec<FeatMap> {
    if !y_axis {
        vec![
            FeatMap {
                data: d2x(g, w, h),
                lines: h,
                extent: w,
                weight: 1.0,
            },
            FeatMap {
                data: d1x(gq, w, h),
                lines: h,
                extent: w,
                weight: 0.7,
            },
        ]
    } else {
        let gt = transpose(g, w, h);
        let gqt = transpose(gq, w, h);
        vec![
            FeatMap {
                data: d2x(&gt, h, w),
                lines: w,
                extent: h,
                weight: 1.0,
            },
            FeatMap {
                data: d1x(&gqt, h, w),
                lines: w,
                extent: h,
                weight: 0.7,
            },
        ]
    }
}

/// Feature maps + per-axis estimates, reusable by the full-mode arbiter.
pub struct Pre {
    pub maps_x: Vec<FeatMap>,
    pub maps_y: Vec<FeatMap>,
    pub ex: AxisEstimate,
    pub ey: AxisEstimate,
}

pub fn prepare(rgba: &[u8], w: usize, h: usize) -> Pre {
    let g = to_gray(rgba, w, h);
    let gq = median_quant(&g, w, h);
    let (maps_x, maps_y) = rayon::join(
        || build_maps(&g, &gq, w, h, false),
        || build_maps(&g, &gq, w, h, true),
    );
    let (ex, ey) = rayon::join(
        || axis_estimate(&maps_x, w, &mut FftPlanner::new()),
        || axis_estimate(&maps_y, h, &mut FftPlanner::new()),
    );
    Pre {
        maps_x,
        maps_y,
        ex,
        ey,
    }
}

pub fn detect_pre(pre: &Pre, w: usize, h: usize) -> Detection {
    let mut planner = FftPlanner::new();
    detect_from_estimates(
        &pre.maps_x,
        &pre.maps_y,
        w,
        h,
        &pre.ex,
        &pre.ey,
        &mut planner,
    )
}

pub fn detect(rgba: &[u8], w: usize, h: usize) -> Detection {
    let pre = prepare(rgba, w, h);
    detect_pre(&pre, w, h)
}

pub fn detect_from_estimates(
    maps_x: &[FeatMap],
    maps_y: &[FeatMap],
    w: usize,
    h: usize,
    ex: &AxisEstimate,
    ey: &AxisEstimate,
    planner: &mut FftPlanner<f64>,
) -> Detection {
    let mut sx = ex.step;
    let mut sy = ey.step;
    let (cx, cy) = (&ex.cands, &ey.cands);

    // cross-axis harmonic reconciliation (dither anti-comb suppression)
    let reconcile = |s_small: f64, s_big: f64, ac_small: &[f64]| -> f64 {
        for m in [2f64, 3.0] {
            if (s_big / s_small - m).abs() <= 0.06 * m {
                let pc_big = plain_comb(ac_small, s_big, 16, 8.0);
                let pc_small = plain_comb(ac_small, s_small, 16, 8.0);
                if pc_big >= 0.55 * pc_small && pc_big > 0.0 {
                    return refine_step_acf(ac_small, s_big, 24);
                }
            }
        }
        s_small
    };
    if sx < sy {
        sx = reconcile(sx, sy, &ex.acf);
    } else if sy < sx {
        sy = reconcile(sy, sx, &ey.acf);
    }

    // joint square-ish pairing on strong runner-up agreement
    if (sx / sy).ln().abs() > 0.08 {
        let zx0 = cx.iter().map(|c| c.1).fold(f64::MIN, f64::max);
        let zy0 = cy.iter().map(|c| c.1).fold(f64::MIN, f64::max);
        let mut best_pair: Option<(f64, f64)> = None;
        let mut best_sum = 0f64;
        for &(s1, z1) in cx.iter().take(6) {
            for &(s2, z2) in cy.iter().take(6) {
                if (s1 / s2).ln().abs() <= 0.08
                    && z1 >= 0.6 * zx0
                    && z2 >= 0.6 * zy0
                    && z1 + z2 > best_sum
                {
                    best_sum = z1 + z2;
                    best_pair = Some((s1, s2));
                }
            }
        }
        if let Some((p1, p2)) = best_pair {
            if (p1 / sx).ln().abs() > 1e-6 || (p2 / sy).ln().abs() > 1e-6 {
                sx = refine_step_acf(&ex.acf, p1, 24);
                sy = refine_step_acf(&ey.acf, p2, 24);
            }
        }
    }

    // looser second pass for mild disagreement (5-13%)
    let dis = (sx / sy).ln().abs();
    if dis > 0.05 && dis <= 0.13 {
        let zx0 = cx.iter().map(|c| c.1).fold(f64::MIN, f64::max);
        let zy0 = cy.iter().map(|c| c.1).fold(f64::MIN, f64::max);
        let pull = |cands: &[(f64, f64)], z0: f64, s_other: f64| -> Option<(f64, f64)> {
            let mut best: Option<(f64, f64)> = None;
            for &(s, z) in cands.iter().take(6) {
                if (s / s_other).ln().abs() <= 0.08 && z >= 0.3 * z0 {
                    if best.map_or(true, |b| z > b.1) {
                        best = Some((s, z));
                    }
                }
            }
            best
        };
        let move_y = pull(cy, zy0, sx);
        let move_x = pull(cx, zx0, sy);
        let score_a = zx0 + move_y.map_or(-1e9, |m| m.1);
        let score_b = zy0 + move_x.map_or(-1e9, |m| m.1);
        if let (Some(my), true) = (move_y, move_y.is_some() && score_a >= score_b) {
            let s_new = refine_step_acf(&ey.acf, my.0, 24);
            sy = if (s_new / my.0).ln().abs() < 0.04 {
                s_new
            } else {
                my.0
            };
        } else if let Some(mx) = move_x {
            let s_new = refine_step_acf(&ex.acf, mx.0, 24);
            sx = if (s_new / mx.0).ln().abs() < 0.04 {
                s_new
            } else {
                mx.0
            };
        }
    }

    // drift-aware counting
    let _ = planner;
    let (n_cols, n_rows) = rayon::join(
        || local_count(maps_x, w, sx, &mut FftPlanner::new()),
        || local_count(maps_y, h, sy, &mut FftPlanner::new()),
    );

    let mut cands: Vec<(f64, f64)> = cx.iter().chain(cy.iter()).cloned().collect();
    cands.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    cands.truncate(8);

    Detection {
        step_x: w as f64 / n_cols,
        step_y: h as f64 / n_rows,
        cols: n_cols.round_ties_even().max(1.0) as usize,
        rows: n_rows.round_ties_even().max(1.0) as usize,
        candidates: cands,
    }
}
