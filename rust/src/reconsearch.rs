//! Reconstruction-search ("distillability") channel, trimmed to what
//! core.py's arbitration uses: _prep, AxisData with nbc=1 energies,
//! _s_grid[::3] coarse curves, _trend_fn, eval_s and _score.
//! Mirrors detector/reconsearch.py.

use crate::kmeans::{even_sample, kmeans};
use crate::sigproc::{interp, median};
use rayon::prelude::*;

pub const S_MIN: f64 = 1.6;
pub const S_MAX: f64 = 24.0;
pub const S_RATIO: f64 = 1.025;
pub const MAX_ROWS: usize = 360;
pub const N_ROWBLOCKS: usize = 5;

/// k-means quantize premultiplied RGB (f32 planes), centroid-color image.
fn quantize(rgb: &[f32], n: usize) -> Vec<f32> {
    let sample_idx = even_sample(n, 48_000);
    let sample: Vec<[f32; 3]> = sample_idx
        .iter()
        .map(|&i| [rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]])
        .collect();
    // k limited by unique(round(sample[::7]/8)) colors
    let mut coarse: Vec<[i32; 3]> = sample
        .iter()
        .step_by(7)
        .map(|p| {
            [
                (p[0] as f64 / 8.0).round_ties_even() as i32,
                (p[1] as f64 / 8.0).round_ties_even() as i32,
                (p[2] as f64 / 8.0).round_ties_even() as i32,
            ]
        })
        .collect();
    coarse.sort_unstable();
    coarse.dedup();
    let k = 14usize.min(coarse.len().max(2));
    let centers = kmeans(&sample, k, 25, 0.25, 3, 12345);

    let mut out = vec![0f32; n * 3];
    for i in 0..n {
        let px = [rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]];
        let mut best = 0usize;
        let mut bd = f64::INFINITY;
        for (ci, c) in centers.iter().enumerate() {
            let mut d = 0f64;
            for ch in 0..3 {
                let dd = (px[ch] - c[ch]) as f64;
                d += dd * dd;
            }
            if d < bd {
                bd = d;
                best = ci;
            }
        }
        for ch in 0..3 {
            out[i * 3 + ch] = centers[best][ch];
        }
    }
    out
}

/// 3x3 symmetric eigen-decomposition (Jacobi), eigenvalues descending.
fn eigh3_desc(cov: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut a = cov;
    let mut v = [[0f64; 3]; 3];
    for i in 0..3 {
        v[i][i] = 1.0;
    }
    for _ in 0..50 {
        // largest off-diagonal
        let mut p = 0usize;
        let mut q = 1usize;
        let mut mx = 0f64;
        for i in 0..3 {
            for j in i + 1..3 {
                if a[i][j].abs() > mx {
                    mx = a[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if mx < 1e-14 {
            break;
        }
        let theta = 0.5 * (a[q][q] - a[p][p]) / a[p][q];
        let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
        let c = 1.0 / (t * t + 1.0).sqrt();
        let s = t * c;
        let (app, aqq, apq) = (a[p][p], a[q][q], a[p][q]);
        a[p][p] = app - t * apq;
        a[q][q] = aqq + t * apq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;
        for i in 0..3 {
            if i != p && i != q {
                let (aip, aiq) = (a[i][p], a[i][q]);
                a[i][p] = c * aip - s * aiq;
                a[p][i] = a[i][p];
                a[i][q] = s * aip + c * aiq;
                a[q][i] = a[i][q];
            }
        }
        for i in 0..3 {
            let (vip, viq) = (v[i][p], v[i][q]);
            v[i][p] = c * vip - s * viq;
            v[i][q] = s * vip + c * viq;
        }
    }
    let mut order = [0usize, 1, 2];
    order.sort_by(|&i, &j| a[j][j].partial_cmp(&a[i][i]).unwrap());
    let evals = [
        a[order[0]][order[0]],
        a[order[1]][order[1]],
        a[order[2]][order[2]],
    ];
    let mut evecs = [[0f64; 3]; 3]; // evecs[comp][channel]
    for (oi, &o) in order.iter().enumerate() {
        for ch in 0..3 {
            evecs[oi][ch] = v[ch][o];
        }
    }
    (evals, evecs)
}

/// _prep: premultiply, quantize, PCA top-2 (+alpha channel if it varies).
/// Returns (channels interleaved f32, n_channels).
pub fn prep(rgba: &[u8], w: usize, h: usize) -> (Vec<f32>, usize) {
    let n = w * h;
    let mut rgb = vec![0f32; n * 3];
    let mut alpha = vec![0f32; n];
    for i in 0..n {
        let p = i * 4;
        let a = rgba[p + 3] as f32;
        alpha[i] = a;
        for c in 0..3 {
            rgb[i * 3 + c] = rgba[p + c] as f32 * (a / 255.0);
        }
    }
    let q = quantize(&rgb, n);
    drop(rgb);

    // PCA over the quantized colors
    let mut mu = [0f64; 3];
    for i in 0..n {
        for c in 0..3 {
            mu[c] += q[i * 3 + c] as f64;
        }
    }
    for c in mu.iter_mut() {
        *c /= n as f64;
    }
    let mut cov = [[0f64; 3]; 3];
    for i in 0..n {
        let x = [
            q[i * 3] as f64 - mu[0],
            q[i * 3 + 1] as f64 - mu[1],
            q[i * 3 + 2] as f64 - mu[2],
        ];
        for a in 0..3 {
            for b in 0..3 {
                cov[a][b] += x[a] * x[b];
            }
        }
    }
    for a in 0..3 {
        for b in 0..3 {
            cov[a][b] /= n.max(1) as f64;
        }
    }
    let (_evals, evecs) = eigh3_desc(cov);

    let amean = alpha.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let avar = alpha
        .iter()
        .map(|&v| (v as f64 - amean).powi(2))
        .sum::<f64>()
        / n as f64;
    let use_alpha = avar.sqrt() > 2.0;
    let nc = if use_alpha { 3 } else { 2 };

    let mut ch = vec![0f32; n * nc];
    for i in 0..n {
        let x = [
            q[i * 3] as f64 - mu[0],
            q[i * 3 + 1] as f64 - mu[1],
            q[i * 3 + 2] as f64 - mu[2],
        ];
        for comp in 0..2 {
            let mut acc = 0f64;
            for c in 0..3 {
                acc += x[c] * evecs[comp][c];
            }
            ch[i * nc + comp] = acc as f32;
        }
        if use_alpha {
            ch[i * nc + 2] = (alpha[i] as f64 - amean) as f32;
        }
    }
    (ch, nc)
}

/// Cumsum tables for one axis (rows subsampled), nbc=1 machinery only.
pub struct AxisData {
    _h: usize,
    w: usize,
    c: usize,
    s: Vec<f32>,   // (H, W+1, C) cumsum
    q_t: Vec<f64>, // per-rowblock total sq sums (nbr)
    r_edges: Vec<usize>,
    nbr: usize,
    t_sum: f64,
}

impl AxisData {
    /// ch: interleaved (h, w, nc); y_axis transposes first.
    pub fn new(ch: &[f32], w: usize, h: usize, nc: usize, y_axis: bool) -> AxisData {
        let (rows0, cols) = if y_axis { (w, h) } else { (h, w) };
        let get = |r: usize, col: usize, c: usize| -> f32 {
            if y_axis {
                ch[(col * w + r) * nc + c]
            } else {
                ch[(r * w + col) * nc + c]
            }
        };
        // row subsample: linspace(0, rows0-1, 360).astype(int)
        let rows_idx: Vec<usize> = if rows0 > MAX_ROWS {
            (0..MAX_ROWS)
                .map(|i| (i as f64 * (rows0 as f64 - 1.0) / (MAX_ROWS as f64 - 1.0)) as usize)
                .collect()
        } else {
            (0..rows0).collect()
        };
        let hh = rows_idx.len();

        let mut s = vec![0f32; hh * (cols + 1) * nc];
        let mut row_sq = vec![0f64; hh]; // total sq sum per row (f64 of f32 cumdiff)
        for (ri, &r0) in rows_idx.iter().enumerate() {
            let base = ri * (cols + 1) * nc;
            let mut acc = vec![0f32; nc];
            let mut accq = vec![0f32; nc];
            for col in 0..cols {
                for c in 0..nc {
                    let v = get(r0, col, c);
                    acc[c] += v;
                    accq[c] += v * v;
                    s[base + (col + 1) * nc + c] = acc[c];
                }
            }
            // qcol for seg[1]: diff of Q at edges [0, W] = total sq sum
            row_sq[ri] = accq.iter().map(|&v| v as f64).sum();
        }

        let nbr = std::cmp::max(1, std::cmp::min(N_ROWBLOCKS, hh / 48));
        let r_edges: Vec<usize> = (0..=nbr)
            .map(|i| (hh as f64 * i as f64 / nbr as f64) as usize)
            .collect();
        let mut q_t = vec![0f64; nbr];
        for b in 0..nbr {
            for r in r_edges[b]..r_edges[b + 1] {
                q_t[b] += row_sq[r];
            }
        }
        // normalizer: total variance about per-rowblock means
        let mut t_sum = 0f64;
        for b in 0..nbr {
            let mut en1 = 0f64;
            for r in r_edges[b]..r_edges[b + 1] {
                let base = r * (cols + 1) * nc;
                let mut sq = 0f64;
                for c in 0..nc {
                    let tot = s[base + cols * nc + c] as f64;
                    sq += tot * tot;
                }
                en1 += sq / cols as f64;
            }
            t_sum += (q_t[b] - en1).max(1e-9);
        }
        AxisData {
            _h: hh,
            w: cols,
            c: nc,
            s,
            q_t,
            r_edges,
            nbr,
            t_sum,
        }
    }

    fn n_phase(&self, s: f64, dense: bool) -> usize {
        let base = if dense { 4.0 * s } else { 2.0 * s };
        let n = base.clamp(8.0, 40.0) as usize;
        n & !1
    }

    /// Per-rowblock cell energies for all phases at step s (nbc=1).
    /// Returns (P, nbr) energies.
    fn energy_tiles(&self, s: f64, phases: &[f64]) -> Vec<Vec<f64>> {
        let w = self.w;
        let nc = self.c;
        let j_cells = (w as f64 / s).ceil() as i64 + 2;
        let np_ = phases.len();
        // per (phase, rowblock) energies; parallel over phases
        let per_phase: Vec<Vec<f64>> = phases
            .par_iter()
            .map(|&phase| {
                // boundaries B[j] = clip(phase + s*(j-1), 0, W), j=0..j_cells
                let nb = (j_cells + 1) as usize;
                let mut bpos = vec![0f64; nb];
                let mut i0 = vec![0usize; nb];
                let mut fr = vec![0f32; nb];
                let mut wds = vec![0f32; nb - 1];
                for j in 0..nb {
                    let b = (phase + s * (j as f64 - 1.0)).clamp(0.0, w as f64);
                    bpos[j] = b;
                    let ii = (b.floor()).clamp(0.0, w as f64 - 1.0) as usize;
                    i0[j] = ii;
                    fr[j] = (b - ii as f64) as f32;
                }
                for j in 0..nb - 1 {
                    wds[j] = (bpos[j + 1] - bpos[j]) as f32;
                }
                let mut out = vec![0f64; self.nbr];
                let mut g = vec![0f32; nb * nc];
                for b in 0..self.nbr {
                    let mut block_acc = 0f64;
                    for r in self.r_edges[b]..self.r_edges[b + 1] {
                        let base = r * (w + 1) * nc;
                        for j in 0..nb {
                            for c in 0..nc {
                                let g0 = self.s[base + i0[j] * nc + c];
                                let g1 = self.s[base + (i0[j] + 1) * nc + c];
                                g[j * nc + c] = g0 + (g1 - g0) * fr[j];
                            }
                        }
                        let mut row_en = 0f64;
                        for j in 0..nb - 1 {
                            let wd = wds[j].max(1e-3);
                            let mut sq = 0f32;
                            for c in 0..nc {
                                let bs = g[(j + 1) * nc + c] - g[j * nc + c];
                                sq += bs * bs;
                            }
                            row_en += (sq / wd) as f64;
                        }
                        // numpy sums the (H,P,J) f32 energies over J in f32
                        block_acc += row_en as f32 as f64;
                    }
                    out[b] = block_acc;
                }
                out
            })
            .collect();
        let _ = np_;
        per_phase
    }

    /// (e_best, e_anti) at step s, per-rowblock best phase (nbc=1).
    pub fn eval_s(&self, s: f64, dense: bool) -> (f64, f64) {
        let n = self.n_phase(s, dense);
        let phases: Vec<f64> = (0..n).map(|i| i as f64 * (s / n as f64)).collect();
        let ens = self.energy_tiles(s, &phases); // (P, nbr)
        let mut err_b = 0f64;
        let mut err_r = 0f64;
        for b in 0..self.nbr {
            let mut ib = 0usize;
            for p in 0..n {
                if ens[p][b] > ens[ib][b] {
                    ib = p;
                }
            }
            let best = ens[ib][b];
            let anti = ens[(ib + n / 2) % n][b];
            err_b += (self.q_t[b] - best).max(0.0);
            err_r += (self.q_t[b] - anti).max(0.0);
        }
        (err_b / self.t_sum, err_r / self.t_sum)
    }
}

/// _s_grid(size): geometric 1.6 .. min(24, size/8), ratio 1.025.
pub fn s_grid(size: usize) -> Vec<f64> {
    let smax = S_MAX.min(size as f64 / 8.0);
    let n = ((smax / S_MIN).ln() / S_RATIO.ln()).ceil() as usize;
    (0..=n).map(|i| S_MIN * S_RATIO.powi(i as i32)).collect()
}

/// _trend_fn data: running median of coarse error vs log-step.
pub struct Trend {
    ls: Vec<f64>,
    trend: Vec<f64>,
}

impl Trend {
    pub fn new(s_list: &[f64], eb: &[f64]) -> Trend {
        let ls: Vec<f64> = s_list.iter().map(|&v| v.ln()).collect();
        let mut trend = vec![0f64; eb.len()];
        for i in 0..eb.len() {
            let sel: Vec<f64> = (0..eb.len())
                .filter(|&j| (ls[j] - ls[i]).abs() <= 0.14)
                .map(|j| eb[j])
                .collect();
            trend[i] = median(&sel).max(1e-6);
        }
        Trend { ls, trend }
    }

    pub fn at(&self, s: f64) -> f64 {
        interp(s.ln(), &self.ls, &self.trend)
    }
}

/// Distillability score.
pub fn score(eb: f64, er: f64, trend_e: f64) -> f64 {
    (er - eb).max(0.0) * (1.0 - eb / trend_e).max(0.0)
}

/// Coarse curves over a step list (parallel).
pub fn coarse_curves(ad: &AxisData, s_list: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let pairs: Vec<(f64, f64)> = s_list.par_iter().map(|&s| ad.eval_s(s, false)).collect();
    (
        pairs.iter().map(|p| p.0).collect(),
        pairs.iter().map(|p| p.1).collect(),
    )
}
