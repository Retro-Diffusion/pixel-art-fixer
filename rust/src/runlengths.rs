//! Grid detection from boundary-run statistics + robust soft-GCD lattice fit.
//! Mirrors detector/runlengths.py exactly (see that file for the theory).
//!
//! Parallelism note: rayon is used only as ordered map -> sequential reduce,
//! so every float accumulation happens in the reference order and results
//! stay bit-identical to the single-threaded path.

use rayon::prelude::*;

pub const S_MIN: f64 = 2.05;
pub const S_MAX: f64 = 26.0;
pub const RUN_MIN: f64 = 2.0;
pub const RUN_MAX: f64 = 64.0;
pub const BIN: f64 = 0.25;
pub const COHERENCE: usize = 7;
pub const THR_FRAC: f32 = 0.10;
pub const MAX_LAG: usize = 4;
pub const TILINGS: [(usize, usize); 3] = [(3, 3), (5, 5), (1, 8)];

/// 3x3 median blur of one u8 channel, border replicate (cv2.medianBlur).
fn median3_channel(src: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let mut buf = [0u8; 9];
        for x in 0..w {
            let mut k = 0;
            for dy in -1i64..=1 {
                let yy = (y as i64 + dy).clamp(0, h as i64 - 1) as usize;
                for dx in -1i64..=1 {
                    let xx = (x as i64 + dx).clamp(0, w as i64 - 1) as usize;
                    buf[k] = src[yy * w + xx];
                    k += 1;
                }
            }
            buf.sort_unstable();
            row[x] = buf[4];
        }
    });
    out
}

/// 3x3 median blur of the 4-channel image (cv2.medianBlur semantics),
/// interleaved u8 out.
pub fn prep_u8(rgba: &[u8], w: usize, h: usize) -> Vec<u8> {
    let chans: Vec<Vec<u8>> = (0..4usize)
        .into_par_iter()
        .map(|c| {
            let plane: Vec<u8> = (0..w * h).map(|i| rgba[i * 4 + c]).collect();
            median3_channel(&plane, w, h)
        })
        .collect();
    let mut out = vec![0u8; w * h * 4];
    for i in 0..w * h {
        for c in 0..4 {
            out[i * 4 + c] = chans[c][i];
        }
    }
    out
}

/// Median-filtered float image, alpha folded in as a 4th channel.
/// Returns (H, W, 4) interleaved f32.
pub fn prep(rgba: &[u8], w: usize, h: usize) -> Vec<f32> {
    prep_u8(rgba, w, h).iter().map(|&v| v as f32).collect()
}

fn transpose4(img: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0f32; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            let s = (y * w + x) * 4;
            let d = (x * h + y) * 4;
            out[d..d + 4].copy_from_slice(&img[s..s + 4]);
        }
    }
    out
}

/// numpy-style linear percentile of the positive entries of `d`.
/// Order statistics via O(n) selection (same values a full sort would give).
fn percentile95_positive(d: &[f32]) -> f32 {
    let mut v: Vec<f32> = d.iter().cloned().filter(|&x| x > 0.0).collect();
    if v.is_empty() {
        return 0.0;
    }
    let n = v.len();
    let pos = 0.95 * (n as f64 - 1.0);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let f = (pos - lo as f64) as f32;
    let (_, &mut vlo, rest) = v.select_nth_unstable_by(lo, |a, b| a.total_cmp(b));
    let vhi = if hi == lo {
        vlo
    } else {
        // hi == lo + 1: the minimum of the right partition
        rest.iter().cloned().fold(f32::INFINITY, f32::min)
    };
    vlo + (vhi - vlo) * f
}

/// Sub-pixel boundary positions along the scan axis of a pre-oriented
/// (rows x cols x 4) image. Returns (scanline_idx, position), row-major.
pub fn boundaries(img4: &[f32], rows: usize, cols: usize) -> (Vec<i64>, Vec<f64>) {
    if cols < 2 {
        return (Vec::new(), Vec::new());
    }
    let dw = cols - 1;
    // L1 color+alpha difference between x-neighbors
    let mut d = vec![0f32; rows * dw];
    d.par_chunks_mut(dw).enumerate().for_each(|(y, row)| {
        for x in 0..dw {
            let a = (y * cols + x) * 4;
            let b = a + 4;
            let mut s = 0f32;
            for c in 0..4 {
                s += (img4[b + c] - img4[a + c]).abs();
            }
            row[x] = s;
        }
    });
    // coherence: vertical box mean over COHERENCE rows, border replicate
    let mut sm = vec![0f32; rows * dw];
    let half = (COHERENCE / 2) as i64;
    {
        let d_ref = &d;
        sm.par_chunks_mut(dw).enumerate().for_each(|(y, row)| {
            for x in 0..dw {
                let mut acc = 0f64;
                for k in -half..=half {
                    let yy = (y as i64 + k).clamp(0, rows as i64 - 1) as usize;
                    acc += d_ref[yy * dw + x] as f64;
                }
                row[x] = (acc / COHERENCE as f64) as f32;
            }
        });
    }
    let d = sm;
    let p95 = percentile95_positive(&d);
    let thr = (20.0f32).max(THR_FRAC * p95);

    let mut ys: Vec<i64> = Vec::new();
    let mut pos: Vec<f64> = Vec::new();
    for y in 0..rows {
        let r = y * dw;
        for x in 0..dw {
            let dc = d[r + x];
            let left = if x == 0 { 0.0 } else { d[r + x - 1] };
            let right = if x == dw - 1 { 0.0 } else { d[r + x + 1] };
            if dc > thr && dc > left && dc >= right {
                let dl = d[r + x.saturating_sub(1)];
                let dr = d[r + (x + 1).min(dw - 1)];
                let denom = dl - 2.0 * dc + dr;
                let off = if denom.abs() > 1e-6 {
                    (0.5 * (dl - dr) / denom).clamp(-0.5, 0.5)
                } else {
                    0.0
                };
                ys.push(y as i64);
                pos.push(x as f64 + off as f64);
            }
        }
    }
    if pos.len() < 4 {
        return (Vec::new(), Vec::new());
    }
    (ys, pos)
}

/// Pooled pos[i+lag]-pos[i] (lag 1..MAX_LAG) within each scanline,
/// filtered to [RUN_MIN, RUN_MAX], stored as f32 like the reference.
pub fn lag_diffs(ys: &[i64], pos: &[f64], max_lag: usize) -> Vec<f32> {
    let mut out: Vec<f32> = Vec::new();
    let n = pos.len();
    for lag in 1..=max_lag {
        if n <= lag {
            break;
        }
        for i in 0..n - lag {
            if ys[i + lag] == ys[i] {
                let d = pos[i + lag] - pos[i];
                if d >= RUN_MIN && d <= RUN_MAX {
                    out.push(d as f32);
                }
            }
        }
    }
    out
}

/// numpy-equivalent uniform histogram over (0, RUN_MAX); returns only the
/// non-empty bins as (count, center) pairs.
fn hist(runs: &[f32], bin_w: f64) -> (Vec<f64>, Vec<f64>) {
    let nb = (RUN_MAX / bin_w) as usize + 1;
    let mut counts = vec![0u64; nb];
    for &r in runs {
        let v = r as f64;
        if !(0.0..=RUN_MAX).contains(&v) {
            continue;
        }
        let mut i = (v / RUN_MAX * nb as f64) as usize;
        if i >= nb {
            i = nb - 1;
        }
        // numpy's float-error correction against the exact edges
        let edge = |j: usize| RUN_MAX * j as f64 / nb as f64;
        if v < edge(i) && i > 0 {
            i -= 1;
        } else if i + 1 < nb && v >= edge(i + 1) {
            i += 1;
        }
        counts[i] += 1;
    }
    let mut hv: Vec<f64> = Vec::new();
    let mut cv: Vec<f64> = Vec::new();
    for i in 0..nb {
        if counts[i] > 0 {
            hv.push(counts[i] as f64);
            let e0 = RUN_MAX * i as f64 / nb as f64;
            let e1 = RUN_MAX * (i + 1) as f64 / nb as f64;
            cv.push(0.5 * (e0 + e1));
        }
    }
    (hv, cv)
}

/// S(s) = weighted mean over distances of cos(2*pi*r/s). Returns (S, total).
fn comb_score_grid(runs: &[f32], s_grid: &[f64], bin_w: f64) -> (Vec<f64>, f64) {
    if runs.is_empty() {
        return (vec![0.0; s_grid.len()], 0.0);
    }
    let (hv, cv) = hist(runs, bin_w);
    let total: f64 = hv.iter().sum();
    let wsum = total;
    let s_out: Vec<f64> = s_grid
        .par_iter()
        .map(|&s| {
            let mut acc = 0f64;
            for (j, &c) in cv.iter().enumerate() {
                acc += hv[j] * (2.0 * std::f64::consts::PI * c / s).cos();
            }
            acc / wsum
        })
        .collect();
    (s_out, total)
}

pub struct PickResult {
    pub s: Option<f64>,
    pub v: f64,
    pub cands: Vec<(f64, f64)>,
}

/// Best step from the comb score, with largest-near-tie divisor logic.
pub fn pick_step(runs: &[f32], s_grid: &[f64]) -> PickResult {
    let none = PickResult {
        s: None,
        v: 0.0,
        cands: Vec::new(),
    };
    let (s_score, total) = comb_score_grid(runs, s_grid, BIN);
    if total < 50.0 {
        return none;
    }
    let ns = s_grid.len();
    let mut idx: Vec<usize> = Vec::new();
    for i in 1..ns.saturating_sub(1) {
        if s_score[i] > s_score[i - 1] && s_score[i] >= s_score[i + 1] {
            idx.push(i);
        }
    }
    if idx.is_empty() {
        return none;
    }
    let mut order = idx.clone();
    order.sort_by(|&a, &b| s_score[b].partial_cmp(&s_score[a]).unwrap());
    let smax = s_score[order[0]];
    if smax <= 0.0 {
        return none;
    }
    let cands: Vec<(f64, f64)> = order
        .iter()
        .take(12)
        .map(|&i| (s_grid[i], s_score[i]))
        .collect();

    let fund = |s: f64| -> f64 {
        let tol = (0.6f64).max(0.18 * s);
        let cnt = runs
            .iter()
            .filter(|&&r| ((r as f64) - s).abs() < tol)
            .count();
        cnt as f64 / total
    };

    let mut tied: Vec<(f64, f64)> = order
        .iter()
        .filter(|&&i| s_score[i] >= 0.70 * smax)
        .map(|&i| (s_grid[i], s_score[i]))
        .collect();
    tied.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let mut best_s = s_grid[order[0]];
    let mut best_v = smax;
    for &(s, v) in &tied {
        if fund(s) >= 0.04 {
            best_s = s;
            best_v = v;
            break;
        }
    }
    PickResult {
        s: Some(best_s),
        v: best_v,
        cands,
    }
}

fn arange(start: f64, stop: f64, step: f64) -> Vec<f64> {
    let n = ((stop - start) / step).ceil().max(0.0) as usize;
    (0..n).map(|i| start + i as f64 * step).collect()
}

/// Sub-pixel refinement: fine k-weighted comb + k-weighted LS polish.
pub fn refine(runs: &[f32], s0: f64) -> f64 {
    if runs.is_empty() {
        return s0;
    }
    let (hv, cv) = hist(runs, 0.05);
    let fine = arange(0.94 * s0, 1.06 * s0, 0.002);
    let sf: Vec<f64> = fine
        .par_iter()
        .map(|&f| {
            let mut acc = 0f64;
            for j in 0..cv.len() {
                let wk = cv[j] / s0;
                acc += hv[j] * wk * (2.0 * std::f64::consts::PI * cv[j] / f).cos();
            }
            acc
        })
        .collect();
    let mut best_i = 0;
    for fi in 0..sf.len() {
        if sf[fi] > sf[best_i] {
            best_i = fi;
        }
    }
    let mut s = fine[best_i];
    for _ in 0..2 {
        let mut num = 0f64;
        let mut den = 0f64;
        for &r in runs {
            let rf = r as f64;
            let k = (rf / s).round_ties_even();
            if k < 1.0 {
                continue;
            }
            let res = (rf - k * s).abs();
            let w = (1.0 - res / (0.30 * s)).clamp(0.0, 1.0) * k;
            num += w * k * rf;
            den += w * k * k;
        }
        if den <= 0.0 {
            break;
        }
        s = num / den;
    }
    s
}

/// Fine comb peak near s0 for one tile; None if unreliable.
fn tile_peak(diffs: &[f32], s0: f64) -> Option<f64> {
    if diffs.len() < 350 {
        return None;
    }
    let fine = arange(0.87 * s0, 1.13 * s0, 0.005);
    let (hv, cv) = hist(diffs, 0.05);
    let total: f64 = hv.iter().sum();
    let sf: Vec<f64> = fine
        .iter()
        .map(|&f| {
            let mut acc = 0f64;
            for j in 0..cv.len() {
                acc += hv[j] * (2.0 * std::f64::consts::PI * cv[j] / f).cos();
            }
            acc / total
        })
        .collect();
    let mut best_i = 0;
    for fi in 0..sf.len() {
        if sf[fi] > sf[best_i] {
            best_i = fi;
        }
    }
    if best_i == 0 || best_i == fine.len() - 1 || sf[best_i] < 0.12 {
        return None;
    }
    Some(fine[best_i])
}

/// cols = W * mean(1/s_local), pooled over several tile grids.
pub fn integrate_step(ys: &[i64], pos: &[f64], n_perp: usize, n_scan: usize, s0: f64) -> f64 {
    // enumerate tiles in reference order, evaluate in parallel, reduce in order
    let mut tiles: Vec<(f64, f64, f64, f64)> = Vec::new();
    for &(tp, tsc) in TILINGS.iter() {
        // np.linspace: value = i * (extent / n)
        let dy = n_perp as f64 / tp as f64;
        let dx = n_scan as f64 / tsc as f64;
        for i in 0..tp {
            let y0 = i as f64 * dy;
            let y1 = if i + 1 == tp {
                n_perp as f64
            } else {
                (i + 1) as f64 * dy
            };
            for j in 0..tsc {
                let x0 = j as f64 * dx;
                let x1 = if j + 1 == tsc {
                    n_scan as f64
                } else {
                    (j + 1) as f64 * dx
                };
                tiles.push((y0, y1, x0, x1));
            }
        }
    }
    let peaks: Vec<Option<f64>> = tiles
        .par_iter()
        .map(|&(y0, y1, x0, x1)| {
            let mut tys: Vec<i64> = Vec::new();
            let mut tpos: Vec<f64> = Vec::new();
            for k in 0..pos.len() {
                let yf = ys[k] as f64;
                if yf >= y0 && yf < y1 && pos[k] >= x0 && pos[k] < x1 {
                    tys.push(ys[k]);
                    tpos.push(pos[k]);
                }
            }
            let diffs = lag_diffs(&tys, &tpos, MAX_LAG);
            tile_peak(&diffs, s0)
        })
        .collect();
    let inv: Vec<f64> = peaks.iter().flatten().map(|&s| 1.0 / s).collect();
    if inv.is_empty() {
        return s0;
    }
    let mean_inv = inv.iter().sum::<f64>() / inv.len() as f64;
    1.0 / mean_inv
}

pub struct RlDetection {
    pub step_x: f64,
    pub step_y: f64,
    pub cols: i64,
    pub rows: i64,
    pub score_x: f64,
    pub score_y: f64,
    pub nruns_x: usize,
    pub nruns_y: usize,
    pub candidates: Vec<(f64, f64)>,
}

struct AxisData {
    s: Option<f64>,
    v: f64,
    cands: Vec<(f64, f64)>,
    runs: Vec<f32>,
    ys: Vec<i64>,
    pos: Vec<f64>,
}

fn axis_pass(img4: &[f32], rows: usize, cols: usize, s_grid: &[f64]) -> AxisData {
    let (ys, pos) = boundaries(img4, rows, cols);
    let runs = lag_diffs(&ys, &pos, MAX_LAG);
    let picked = pick_step(&runs, s_grid);
    let s = picked.s.map(|s| refine(&runs, s));
    AxisData {
        s,
        v: picked.v,
        cands: picked.cands,
        runs,
        ys,
        pos,
    }
}

pub fn detect(rgba: &[u8], w: usize, h: usize) -> RlDetection {
    let img4 = prep(rgba, w, h);
    let s_grid = arange(S_MIN, S_MAX, 0.01);

    let img4t = transpose4(&img4, w, h);
    let (ax, ay) = rayon::join(
        || axis_pass(&img4, h, w, &s_grid),
        || axis_pass(&img4t, w, h, &s_grid),
    );
    drop(img4t);

    let (mut sx, mut sy) = (ax.s, ay.s);
    let (vx, vy) = (ax.v, ay.v);
    if sx.is_none() && sy.is_none() {
        return RlDetection {
            step_x: 8.0,
            step_y: 8.0,
            cols: (w as f64 / 8.0).round_ties_even() as i64,
            rows: (h as f64 / 8.0).round_ties_even() as i64,
            score_x: vx,
            score_y: vy,
            nruns_x: ax.runs.len(),
            nruns_y: ay.runs.len(),
            candidates: Vec::new(),
        };
    }
    // cross-axis reconciliation: a weak axis borrows the strong axis's step
    let borrow_x = match (sx, sy) {
        (None, Some(_)) => true,
        (Some(x), Some(y)) => vx < 0.5 * vy && (x - y).abs() > 0.15 * y,
        _ => false,
    };
    if borrow_x {
        let syv = sy.unwrap();
        let sx2 = if !ax.runs.is_empty() {
            refine(&ax.runs, syv)
        } else {
            syv
        };
        if (sx2 - syv).abs() < 0.15 * syv {
            sx = Some(sx2);
        }
    }
    let borrow_y = match (sy, sx) {
        (None, Some(_)) => true,
        (Some(y), Some(x)) => vy < 0.5 * vx && (y - x).abs() > 0.15 * x,
        _ => false,
    };
    if borrow_y {
        let sxv = sx.unwrap();
        let sy2 = if !ay.runs.is_empty() {
            refine(&ay.runs, sxv)
        } else {
            sxv
        };
        if (sy2 - sxv).abs() < 0.15 * sxv {
            sy = Some(sy2);
        }
    }
    let mut sx = sx.unwrap_or_else(|| sy.unwrap());
    let mut sy = sy.unwrap_or(sx);

    // local-step integration (drift-aware effective step)
    let (sx2, sy2) = rayon::join(
        || integrate_step(&ax.ys, &ax.pos, h, w, sx),
        || integrate_step(&ay.ys, &ay.pos, w, h, sy),
    );
    sx = sx2;
    sy = sy2;

    // square-cell reconciliation: pool near-agreeing axes (harmonic mean)
    let rel = (sx - sy).abs() / (0.5 * (sx + sy));
    if rel < 0.085 || (rel < 0.15 && vx.max(vy) < 0.15) {
        let hm = 2.0 / (1.0 / sx + 1.0 / sy);
        sx = hm;
        sy = hm;
    }

    let mut cands: Vec<(f64, f64)> = ax.cands.iter().chain(ay.cands.iter()).cloned().collect();
    cands.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    RlDetection {
        step_x: sx,
        step_y: sy,
        cols: (w as f64 / sx).round_ties_even() as i64,
        rows: (h as f64 / sy).round_ties_even() as i64,
        score_x: vx,
        score_y: vy,
        nruns_x: ax.runs.len(),
        nruns_y: ay.runs.len(),
        candidates: cands,
    }
}
