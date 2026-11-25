//! Shift self-similarity grid detection. Mirrors detector/selfsim.py.
//!
//! The "quant" feature is weight-0 in the reference (ablated) and is not
//! ported. `_phase` is skipped too: no caller consumes selfsim's phase.

use rayon::prelude::*;

pub const TMAX: usize = 72;
pub const PMIN: f64 = 1.7;
pub const PMAX: f64 = 24.0;
pub const PSTEP: f64 = 0.02;
pub const KCAP: usize = 24;
pub const PIX_BUDGET: usize = 1_200_000;
pub const TILE_TARGET: f64 = 144.0;
pub const QUALIFY_FRAC: f64 = 0.40;
pub const QUALIFY_ABS: f64 = 4.0;
pub const VOTE_WIN: f64 = 0.15;
pub const VOTE_MIN: f64 = 1.5;
pub const DISP_UNIFORM: f64 = 0.015;

pub struct Plane {
    pub data: Vec<f32>,
    pub w: usize,
    pub h: usize,
}

impl Plane {
    fn at(&self, y: usize, x: usize) -> f32 {
        self.data[y * self.w + x]
    }
}

fn luma(rgba: &[u8], w: usize, h: usize) -> Plane {
    let mut data = vec![0f32; w * h];
    for i in 0..w * h {
        let p = i * 4;
        data[i] = rgba[p] as f32 * 0.299 + rgba[p + 1] as f32 * 0.587 + rgba[p + 2] as f32 * 0.114;
    }
    Plane { data, w, h }
}

/// reflect101 index (OpenCV BORDER_DEFAULT)
fn refl(i: i64, n: i64) -> usize {
    let mut i = i;
    loop {
        if i < 0 {
            i = -i;
        } else if i >= n {
            i = 2 * n - 2 - i;
        } else {
            return i as usize;
        }
    }
}

/// cv2.GaussianBlur(y, (0,0), 1.0): 9-tap separable Gaussian, reflect101.
fn gaussian_blur_1(src: &Plane) -> Plane {
    let (w, h) = (src.w, src.h);
    // OpenCV kernel: exp(-x^2/(2 sigma^2)) at x = i - 4, normalized (f64),
    // then applied as f32 coefficients
    let mut k64 = [0f64; 9];
    let mut sum = 0f64;
    for i in 0..9 {
        let x = i as f64 - 4.0;
        k64[i] = (-0.5 * x * x).exp();
        sum += k64[i];
    }
    let k: Vec<f32> = k64.iter().map(|&v| (v / sum) as f32).collect();

    let mut tmp = vec![0f32; w * h];
    tmp.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let mut acc = 0f32;
            for t in 0..9 {
                let xx = refl(x as i64 + t as i64 - 4, w as i64);
                acc += k[t] * src.at(y, xx);
            }
            row[x] = acc;
        }
    });
    let mut out = vec![0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let mut acc = 0f32;
            for t in 0..9 {
                let yy = refl(y as i64 + t as i64 - 4, h as i64);
                acc += k[t] * tmp[yy * w + x];
            }
            row[x] = acc;
        }
    });
    Plane { data: out, w, h }
}

/// |cv2.Laplacian(x, CV_32F)| (3x3 aperture, reflect101).
fn abs_laplacian(src: &Plane) -> Plane {
    let (w, h) = (src.w, src.h);
    let mut out = vec![0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let up = src.at(refl(y as i64 - 1, h as i64), x);
            let dn = src.at(refl(y as i64 + 1, h as i64), x);
            let lf = src.at(y, refl(x as i64 - 1, w as i64));
            let rt = src.at(y, refl(x as i64 + 1, w as i64));
            let c = src.at(y, x);
            row[x] = (up + dn + lf + rt - 4.0 * c).abs();
        }
    });
    Plane { data: out, w, h }
}

/// |d/dx| + |d/dy| with prepend-first semantics (first row/col diff = 0).
fn gradmag(y: &Plane) -> Plane {
    let (w, h) = (y.w, y.h);
    let mut out = vec![0f32; w * h];
    for r in 0..h {
        for c in 0..w {
            let gx = if c == 0 {
                0.0
            } else {
                (y.at(r, c) - y.at(r, c - 1)).abs()
            };
            let gy = if r == 0 {
                0.0
            } else {
                (y.at(r, c) - y.at(r - 1, c)).abs()
            };
            out[r * w + c] = gx + gy;
        }
    }
    Plane { data: out, w, h }
}

pub struct Features {
    pub grad0: Plane,
    pub gradb: Plane,
    pub lapb: Plane,
}

pub fn build_features(rgba: &[u8], w: usize, h: usize) -> (Features, Vec<f32>) {
    let y = luma(rgba, w, h);
    let yb = gaussian_blur_1(&y);
    let grad0 = gradmag(&y);
    let gradb = gradmag(&yb);
    let lapb = abs_laplacian(&yb);
    let mut wt = vec![0f32; w * h];
    for i in 0..w * h {
        wt[i] = (rgba[i * 4 + 3] as f32 / 255.0).max(0.05);
    }
    (Features { grad0, gradb, lapb }, wt)
}

fn transpose_plane(p: &Plane) -> Plane {
    let mut data = vec![0f32; p.w * p.h];
    for y in 0..p.h {
        for x in 0..p.w {
            data[x * p.h + y] = p.data[y * p.w + x];
        }
    }
    Plane {
        data,
        w: p.h,
        h: p.w,
    }
}

/// np.unique(np.linspace(0, n, parts+1)[:-1].astype(int))
fn tile_starts(n: usize, parts: usize) -> Vec<usize> {
    let step = n as f64 / parts as f64;
    let mut v: Vec<usize> = (0..parts).map(|i| (i as f64 * step) as usize).collect();
    v.dedup();
    v
}

/// Per-tile numerator/denominator d(t) curves for one (pre-oriented)
/// single-channel feature. Returns (num, den) shaped [n_tiles_y][n_tiles_x][T].
#[allow(clippy::type_complexity)]
fn dcurves_tiled(
    arr0: &Plane,
    wt0: &Plane,
    tmax: usize,
    ny: usize,
    nx: usize,
) -> (Vec<Vec<Vec<f64>>>, Vec<Vec<Vec<f64>>>, usize) {
    let (full_h, w) = (arr0.h, arr0.w);
    let stride = std::cmp::max(1, (full_h * w + PIX_BUDGET - 1) / PIX_BUDGET);
    let rows_idx: Vec<usize> = (0..full_h).step_by(stride).collect();
    let h = rows_idx.len();
    let t_cap = std::cmp::max(std::cmp::min(tmax, w / 3), 1);

    let rs = tile_starts(h, ny);
    let cs0 = tile_starts(w, nx);
    let (ny, nx) = (rs.len(), cs0.len());

    let mut num = vec![vec![vec![0f64; t_cap]; nx]; ny];
    let mut den = vec![vec![vec![0f64; t_cap]; nx]; ny];

    // per-t work is independent; parallelize over t and merge
    let results: Vec<(usize, Vec<Vec<f64>>, Vec<Vec<f64>>, bool)> = (1..=t_cap)
        .into_par_iter()
        .map(|t| {
            // column boundaries clamped like the reference
            let clamp_at = if w >= t + 1 { w - t - 1 } else { 0 };
            let cs: Vec<usize> = cs0.iter().map(|&c| c.min(clamp_at)).collect();
            let mut cs_u = cs.clone();
            cs_u.dedup();
            let degenerate = cs_u.len() != nx;
            let use_cs: Vec<usize> = if degenerate { vec![0] } else { cs };
            let ncols = use_cs.len();
            let ext = w - t;

            // two-stage sums like the reference reduceat chain: rows first
            // (f32-rounded per column), then column segments (f32-rounded)
            let mut nu = vec![vec![0f64; ncols]; ny];
            let mut de = vec![vec![0f64; ncols]; ny];
            let mut coln = vec![0f64; ext];
            let mut cold = vec![0f64; ext];
            for (ri, &r0) in rs.iter().enumerate() {
                let r1 = if ri + 1 < ny { rs[ri + 1] } else { h };
                coln.iter_mut().for_each(|v| *v = 0.0);
                cold.iter_mut().for_each(|v| *v = 0.0);
                for rr in r0..r1 {
                    let y = rows_idx[rr];
                    let arow = &arr0.data[y * w..y * w + w];
                    let wrow = &wt0.data[y * w..y * w + w];
                    for x in 0..ext {
                        let diff = (arow[x + t] - arow[x]).abs();
                        let wp = wrow[x + t] * wrow[x];
                        coln[x] += (diff * wp) as f64;
                        cold[x] += wp as f64;
                    }
                }
                for (ci, &c0) in use_cs.iter().enumerate() {
                    let c1 = if ci + 1 < ncols { use_cs[ci + 1] } else { ext };
                    let mut anum = 0f64;
                    let mut aden = 0f64;
                    for x in c0..c1.max(c0) {
                        anum += coln[x] as f32 as f64;
                        aden += cold[x] as f32 as f64;
                    }
                    nu[ri][ci] = anum as f32 as f64;
                    de[ri][ci] = aden as f32 as f64;
                }
            }
            (t, nu, de, degenerate)
        })
        .collect();

    for (t, nu, de, degenerate) in results {
        if degenerate && nx > 1 {
            let nsum: f64 = nu.iter().flatten().sum();
            let dsum: f64 = de.iter().flatten().sum();
            for a in 0..ny {
                for b in 0..nx {
                    num[a][b][t - 1] += nsum / (ny * nx) as f64;
                    den[a][b][t - 1] += dsum / (ny * nx) as f64;
                }
            }
        } else {
            for a in 0..ny {
                for b in 0..nx.min(nu[a].len()) {
                    num[a][b][t - 1] = nu[a][b];
                    den[a][b][t - 1] = de[a][b];
                }
            }
        }
    }
    (num, den, t_cap)
}

/// Regroup tile num/den grids into gy x gx summed d(t) curves.
fn group(num: &[Vec<Vec<f64>>], den: &[Vec<Vec<f64>>], gy: usize, gx: usize) -> Vec<Vec<f64>> {
    let ny = num.len();
    let nx = num[0].len();
    let t_len = num[0][0].len();
    let ys: Vec<usize> = (0..=gy)
        .map(|i| (ny as f64 * i as f64 / gy as f64) as usize)
        .collect();
    let xs: Vec<usize> = (0..=gx)
        .map(|i| (nx as f64 * i as f64 / gx as f64) as usize)
        .collect();
    let mut out: Vec<Vec<f64>> = Vec::with_capacity(gy * gx);
    for a in 0..gy {
        for b in 0..gx {
            let mut n = vec![0f64; t_len];
            let mut d = vec![0f64; t_len];
            for yy in ys[a]..ys[a + 1] {
                for xx in xs[b]..xs[b + 1] {
                    for t in 0..t_len {
                        n[t] += num[yy][xx][t];
                        d[t] += den[yy][xx][t];
                    }
                }
            }
            for t in 0..t_len {
                n[t] /= d[t].max(1e-9);
            }
            out.push(n);
        }
    }
    out
}

fn interp1(xs_start: f64, dm: &[f64], q: f64) -> f64 {
    // np.interp over ts = 1..T with clamping
    let t_len = dm.len();
    let pos = q - xs_start;
    if pos <= 0.0 {
        return dm[0];
    }
    if pos >= (t_len - 1) as f64 {
        return dm[t_len - 1];
    }
    let i = pos.floor() as usize;
    let f = pos - i as f64;
    dm[i] * (1.0 - f) + dm[i + 1] * f
}

/// t-statistic comb score for one normalized curve dm(t), t = 1..T.
fn comb_tstat(dm: &[f64], p_grid: &[f64], kcap: usize) -> Vec<f64> {
    let t_len = dm.len();
    let mut s_out = vec![-1e9; p_grid.len()];
    if t_len < 8 {
        return s_out;
    }
    // trapezoid cumulative integral F over ts = 1..T
    let mut cum = vec![0f64; t_len];
    for i in 1..t_len {
        cum[i] = cum[i - 1] + 0.5 * (dm[i] + dm[i - 1]);
    }
    let mut d2: Vec<f64> = (2..t_len)
        .map(|i| (dm[i] - 2.0 * dm[i - 1] + dm[i - 2]).abs())
        .collect();
    let noise = if d2.is_empty() {
        1e-9
    } else {
        d2.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = d2.len();
        (if n % 2 == 1 {
            d2[n / 2]
        } else {
            0.5 * (d2[n / 2 - 1] + d2[n / 2])
        }) + 1e-9
    };
    let boxm = |q: f64, p: f64| -> f64 {
        let lo = (q - p / 2.0).max(1.0);
        let hi = (q + p / 2.0).min(t_len as f64);
        (interp1(1.0, &cum, hi) - interp1(1.0, &cum, lo)) / (hi - lo).max(1e-9)
    };
    for (pi, &p) in p_grid.iter().enumerate() {
        let kf = (t_len as f64 / p - 0.5).floor() as i64;
        let k = std::cmp::min(kf.max(0) as usize, kcap);
        if k < 2 {
            continue;
        }
        let mut c = vec![0f64; k];
        for kk in 1..=k {
            let q = p * kk as f64;
            let rm = interp1(1.0, dm, q) - boxm(q, p);
            let rh1 = interp1(1.0, dm, q - p / 2.0) - boxm(q - p / 2.0, p);
            let rh2 = interp1(1.0, dm, q + p / 2.0) - boxm(q + p / 2.0, p);
            c[kk - 1] = 0.5 * (rh1 + rh2) - rm;
        }
        let mean_c = c.iter().sum::<f64>() / k as f64;
        let var = c.iter().map(|&v| (v - mean_c) * (v - mean_c)).sum::<f64>() / k as f64;
        let std_c = var.sqrt();
        let kpen = (1.0f64).min((k as f64 - 1.0) / 3.0);
        s_out[pi] = kpen * mean_c * (k as f64).sqrt() / (std_c + 0.5 * noise + 1e-9);
    }
    s_out
}

/// local maxima of z (>= left, > right), sorted by z descending
fn local_maxima(z: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = Vec::new();
    for i in 1..z.len().saturating_sub(1) {
        if z[i] >= z[i - 1] && z[i] > z[i + 1] {
            idx.push(i);
        }
    }
    idx.sort_by(|&a, &b| z[b].partial_cmp(&z[a]).unwrap());
    idx
}

fn parabolic(x0: f64, dx: f64, y: &[f64], i: usize) -> f64 {
    if i > 0 && i < y.len() - 1 {
        let denom = y[i - 1] - 2.0 * y[i] + y[i + 1];
        if denom.abs() > 1e-12 {
            let off = (0.5 * (y[i - 1] - y[i + 1]) / denom).clamp(-1.0, 1.0);
            return x0 + i as f64 * dx + off * dx;
        }
    }
    x0 + i as f64 * dx
}

/// Sub-pixel step from minima of R(t) near k*s0, weighted LS fit.
fn refine_step(r: &[f64], s0: f64) -> f64 {
    let t_len = r.len();
    let kmax = std::cmp::min(KCAP, ((t_len as f64 - 1.0) / s0) as usize);
    let mut pairs: Vec<(f64, f64, f64)> = Vec::new();
    for k in 1..=kmax {
        let t0 = k as f64 * s0;
        let half = (1.5f64).max(0.3 * s0);
        let lo = std::cmp::max((t0 - half).floor() as i64, 1) as usize;
        let hi = std::cmp::min((t0 + half).ceil() as usize, t_len);
        if hi < lo + 2 {
            continue;
        }
        let seg = &r[lo - 1..hi];
        let mut j = 0;
        for jj in 0..seg.len() {
            if seg[jj] < seg[j] {
                j = jj;
            }
        }
        let tk = parabolic(lo as f64, 1.0, seg, j);
        pairs.push((k as f64, tk, 0.85f64.powi(k as i32 - 1)));
    }
    if pairs.is_empty() {
        return s0;
    }
    let mut s = s0;
    for _ in 0..2 {
        let num: f64 = pairs.iter().map(|&(k, t, w)| w * k * t).sum();
        let den: f64 = pairs.iter().map(|&(k, _, w)| w * k * k).sum();
        s = num / den;
        let tol = (0.2 * s0).max(1.0);
        let keep: Vec<(f64, f64, f64)> = pairs
            .iter()
            .cloned()
            .filter(|&(k, t, _)| (t - k * s).abs() <= tol)
            .collect();
        if keep.len() == pairs.len() || keep.is_empty() {
            break;
        }
        pairs = keep;
    }
    if 0.6 * s0 < s && s < 1.4 * s0 {
        s
    } else {
        s0
    }
}

/// Mean-normalized, box(9)-detrended residual (edge-padded convolution).
fn residual_curve(d: &[f64]) -> Vec<f64> {
    let n = d.len();
    let mean = d.iter().sum::<f64>() / n.max(1) as f64;
    if mean < 1e-9 {
        return vec![0.0; n];
    }
    let dm: Vec<f64> = d.iter().map(|&v| v / mean).collect();
    let mut out = vec![0f64; n];
    for i in 0..n {
        let mut acc = 0f64;
        for k in -4i64..=4 {
            let j = (i as i64 + k).clamp(0, n as i64 - 1) as usize;
            acc += dm[j];
        }
        out[i] = dm[i] - acc / 9.0;
    }
    out
}

fn score_curves(curves: &[Vec<f64>], p_grid: &[f64], fw: f64, kcap: usize) -> Vec<Vec<f64>> {
    curves
        .par_iter()
        .map(|d| {
            let m = d.iter().sum::<f64>() / d.len().max(1) as f64;
            if m > 1e-9 {
                let dn: Vec<f64> = d.iter().map(|&v| v / m).collect();
                comb_tstat(&dn, p_grid, kcap)
                    .iter()
                    .map(|&v| fw * v.clamp(-10.0, 30.0))
                    .collect()
            } else {
                vec![0.0; p_grid.len()]
            }
        })
        .collect()
}

fn arange(start: f64, stop: f64, step: f64) -> Vec<f64> {
    let n = ((stop - start) / step).ceil().max(0.0) as usize;
    (0..n).map(|i| start + i as f64 * step).collect()
}

/// python round() = banker's rounding
fn pyround(v: f64) -> f64 {
    v.round_ties_even()
}

pub struct SsAxis {
    pub s: f64,
    pub conf: f64,
}

fn detect_axis(feats: [(&Plane, f64); 3], wt: &Plane, size: usize) -> SsAxis {
    let p_grid = arange(PMIN, PMAX.min(size as f64 / 4.0), PSTEP);
    let np_ = p_grid.len();
    // wt here is pre-oriented: shift axis horizontal
    let (h, w) = (wt.h, wt.w);
    let (other, shift) = (h, w);
    let ny = (pyround(other as f64 / 128.0)).clamp(1.0, 8.0) as usize;
    let nx = (pyround(shift as f64 / TILE_TARGET)).clamp(1.0, 6.0) as usize;
    let gy = std::cmp::min(ny, 4);

    let mut s_fine: Vec<Vec<f64>> = Vec::new();
    let mut s_coarse: Vec<Vec<f64>> = Vec::new();
    let mut s_glob = vec![0f64; np_];
    let mut agg_res: Vec<f64> = Vec::new();
    let mut w_fine: Vec<f64> = Vec::new();
    let mut w_coarse: Vec<f64> = Vec::new();
    let mut first = true;

    for (fi, &(plane, fw)) in feats.iter().enumerate() {
        let (num, den, _t) = dcurves_tiled(plane, wt, TMAX, ny, nx);
        let (tny, tnx) = (num.len(), num[0].len());
        let d_fine = group(&num, &den, tny, tnx);
        let d_coarse = group(&num, &den, gy, 1);
        let d_glob = group(&num, &den, 1, 1);
        if first {
            s_fine = vec![vec![0f64; np_]; d_fine.len()];
            s_coarse = vec![vec![0f64; np_]; d_coarse.len()];
            agg_res = vec![0f64; d_glob[0].len()];
            first = false;
        }
        let sf = score_curves(&d_fine, &p_grid, fw, 6);
        for i in 0..d_fine.len() {
            for j in 0..np_ {
                s_fine[i][j] += sf[i][j];
            }
        }
        let sc = score_curves(&d_coarse, &p_grid, fw, KCAP);
        for i in 0..d_coarse.len() {
            for j in 0..np_ {
                s_coarse[i][j] += sc[i][j];
            }
        }
        let sg = score_curves(&d_glob, &p_grid, fw, KCAP);
        for j in 0..np_ {
            s_glob[j] += sg[0][j];
        }
        let res = residual_curve(&d_glob[0]);
        for j in 0..agg_res.len() {
            agg_res[j] += fw * res[j];
        }
        if fi == 0 {
            // grad0 tile weights
            w_fine = d_fine
                .iter()
                .map(|c| c.iter().sum::<f64>() / c.len().max(1) as f64)
                .collect();
            w_coarse = d_coarse
                .iter()
                .map(|c| c.iter().sum::<f64>() / c.len().max(1) as f64)
                .collect();
        }
    }
    let wc_sum: f64 = w_coarse.iter().sum();
    if s_fine.is_empty() || wc_sum <= 1e-9 {
        return SsAxis { s: 8.0, conf: 0.0 };
    }
    let mut stot = vec![0f64; np_];
    for j in 0..np_ {
        let mut acc = 0f64;
        for i in 0..s_coarse.len() {
            acc += (w_coarse[i] / wc_sum) * s_coarse[i][j];
        }
        stot[j] = (acc + 0.5 * s_glob[j]) / 1.5;
    }
    let peaks = local_maxima(&stot);
    if peaks.is_empty() || stot[peaks[0]] <= 0.0 {
        return SsAxis { s: 8.0, conf: 0.0 };
    }
    let smax = stot[peaks[0]];
    let floor = (QUALIFY_FRAC * smax).max(QUALIFY_ABS);
    let qual: Vec<usize> = peaks
        .iter()
        .cloned()
        .filter(|&i| stot[i] >= floor)
        .collect();
    let mut best_i = if qual.is_empty() {
        peaks[0]
    } else {
        *qual
            .iter()
            .min_by(|&&a, &&b| p_grid[a].partial_cmp(&p_grid[b]).unwrap())
            .unwrap()
    };
    // explicit divisor walk
    let mut changed = true;
    while changed {
        changed = false;
        for m in [6f64, 5.0, 4.0, 3.0, 2.0] {
            let pf = p_grid[best_i] / m;
            if pf < p_grid[0] {
                continue;
            }
            let j0 = pyround((pf - p_grid[0]) / PSTEP) as i64;
            let rad = std::cmp::max(pyround(0.05 * pf / PSTEP) as i64, 4);
            let lo = std::cmp::max(j0 - rad, 0) as usize;
            let hi = std::cmp::min((j0 + rad + 1) as usize, np_);
            if hi > lo {
                let mut j = lo;
                for jj in lo..hi {
                    if stot[jj] > stot[j] {
                        j = jj;
                    }
                }
                if stot[j] >= (0.45 * stot[best_i]).max(0.8 * QUALIFY_ABS) {
                    best_i = j;
                    changed = true;
                    break;
                }
            }
        }
    }
    let p_star = p_grid[best_i];

    // per-fine-tile votes around a center
    let tile_votes = |center: f64| -> (Vec<f64>, Vec<f64>) {
        let win = (VOTE_WIN * center).max(6.0 * PSTEP);
        let lo = std::cmp::max(pyround((center - win - p_grid[0]) / PSTEP) as i64, 0) as usize;
        let hi = std::cmp::min(
            pyround((center + win - p_grid[0]) / PSTEP) as i64 + 1,
            np_ as i64,
        ) as usize;
        let mut vs = Vec::new();
        let mut ws = Vec::new();
        for i in 0..s_fine.len() {
            if w_fine[i] <= 1e-9 || hi <= lo || hi - lo < 3 {
                continue;
            }
            let seg = &s_fine[i][lo..hi];
            let mut j = 0;
            for jj in 0..seg.len() {
                if seg[jj] > seg[j] {
                    j = jj;
                }
            }
            if seg[j] < VOTE_MIN {
                continue;
            }
            vs.push(parabolic(p_grid[lo], PSTEP, seg, j));
            ws.push(seg[j].min(8.0));
        }
        (vs, ws)
    };
    let wmedian = |v: &[f64], w: &[f64]| -> f64 {
        let mut order: Vec<usize> = (0..v.len()).collect();
        order.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap());
        let total: f64 = w.iter().sum();
        let mut cw = 0f64;
        for &o in &order {
            cw += w[o];
            if cw >= 0.5 * total {
                return v[o];
            }
        }
        v[order[order.len() - 1]]
    };

    let (mut votes, mut vw) = tile_votes(p_star);
    if !votes.is_empty() {
        let med = wmedian(&votes, &vw);
        let re = tile_votes(med);
        votes = re.0;
        vw = re.1;
    }
    let s0 = if !votes.is_empty() {
        let med = wmedian(&votes, &vw);
        let devs: Vec<f64> = votes.iter().map(|&v| (v - med).abs()).collect();
        let disp = wmedian(&devs, &vw) / med.max(1e-9);
        if disp < DISP_UNIFORM {
            let mut s0 = med;
            let s_ls = refine_step(&agg_res, s0);
            if (s_ls - s0).abs() <= (0.03 * s0).max(0.04) {
                s0 = s_ls;
            }
            s0
        } else {
            let wsum: f64 = vw.iter().sum();
            let hsum: f64 = vw.iter().zip(votes.iter()).map(|(&w, &v)| w / v).sum();
            wsum / hsum
        }
    } else {
        let mut s0 = parabolic(p_grid[0], PSTEP, &stot, best_i);
        let s_ls = refine_step(&agg_res, s0);
        if (s_ls - s0).abs() <= (0.03 * s0).max(0.04) {
            s0 = s_ls;
        }
        s0
    };
    SsAxis {
        s: s0,
        conf: stot[best_i],
    }
}

pub struct SsDetection {
    pub step_x: f64,
    pub step_y: f64,
    pub cols: i64,
    pub rows: i64,
    pub conf: f64,
}

pub fn detect(rgba: &[u8], w: usize, h: usize) -> SsDetection {
    let (feats, wt) = build_features(rgba, w, h);
    let wt_plane = Plane { data: wt, w, h };
    // pre-orient the y-axis pass (shift axis horizontal)
    let g0t = transpose_plane(&feats.grad0);
    let gbt = transpose_plane(&feats.gradb);
    let lbt = transpose_plane(&feats.lapb);
    let wtt = transpose_plane(&wt_plane);

    let (ax, ay) = rayon::join(
        || {
            detect_axis(
                [(&feats.grad0, 1.0), (&feats.gradb, 1.0), (&feats.lapb, 1.0)],
                &wt_plane,
                w,
            )
        },
        || detect_axis([(&g0t, 1.0), (&gbt, 1.0), (&lbt, 1.0)], &wtt, h),
    );
    SsDetection {
        step_x: ax.s,
        step_y: ay.s,
        cols: std::cmp::max(1, pyround(w as f64 / ax.s) as i64),
        rows: std::cmp::max(1, pyround(h as f64 / ay.s) as i64),
        conf: ax.conf.min(ay.conf),
    }
}
