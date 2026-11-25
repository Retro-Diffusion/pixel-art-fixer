//! CellVarContrast "square packer" channel (detector/varcontrast.py),
//! trimmed to what core.py's arbitration reads: the detrended z curve
//! (scored_curve) exposed via z_of(step).

use crate::sigproc::{median, median_filter1d};

fn pyround(v: f64) -> f64 {
    v.round_ties_even()
}

pub struct CellVarContrast {
    w: usize,
    h: usize,
    c: usize,
    scflat: Vec<f64>, // (h+1)*(w+1) x (C+1) fused SAT: channels + sqsum
    total_var: f64,
    active_var: f64,
    px: Vec<f64>,
    py: Vec<f64>,
}

impl CellVarContrast {
    pub fn new(rgba: &[u8], w: usize, h: usize) -> CellVarContrast {
        let c = 4usize;
        // premultiplied f64 image with alpha channel
        let stride = w + 1;
        let mut scflat = vec![0f64; (h + 1) * stride * (c + 1)];
        // build cumulative sums row by row
        {
            let idx = |y: usize, x: usize, ch: usize| (y * stride + x) * (c + 1) + ch;
            for y in 0..h {
                let mut row = vec![0f64; c + 1];
                for x in 0..w {
                    let p = (y * w + x) * 4;
                    let a = rgba[p + 3] as f64 / 255.0;
                    let vals = [
                        rgba[p] as f64 * a,
                        rgba[p + 1] as f64 * a,
                        rgba[p + 2] as f64 * a,
                        rgba[p + 3] as f64,
                    ];
                    let sq: f64 = vals.iter().map(|&v| v * v).sum();
                    for ch in 0..c {
                        row[ch] += vals[ch];
                    }
                    row[c] += sq;
                    for ch in 0..=c {
                        scflat[idx(y + 1, x + 1, ch)] = scflat[idx(y, x + 1, ch)] + row[ch];
                    }
                }
            }
        }
        let n = (h * w) as f64;
        let idx = |y: usize, x: usize, ch: usize| (y * stride + x) * (c + 1) + ch;
        let mut mean_sq = 0f64;
        for ch in 0..c {
            let m = scflat[idx(h, w, ch)] / n;
            mean_sq += m * m;
        }
        let total_var = scflat[idx(h, w, c)] / n - mean_sq;

        // activity map: variance of 8x8 blocks
        let mut bs = 8usize;
        let mut by: Vec<usize> = (0..h.saturating_sub(bs - 1)).step_by(bs).collect();
        let mut bx: Vec<usize> = (0..w.saturating_sub(bs - 1)).step_by(bs).collect();
        if by.is_empty() || bx.is_empty() {
            by = vec![0];
            bx = vec![0];
            bs = h.min(w);
        }
        let area = (bs * bs) as f64;
        let thresh = (0.02 * total_var).max(1e-6);
        let mut act_px: Vec<f64> = Vec::new();
        let mut act_py: Vec<f64> = Vec::new();
        let mut all_px: Vec<f64> = Vec::new();
        let mut all_py: Vec<f64> = Vec::new();
        let mut active_sum = 0f64;
        let mut active_cnt = 0usize;
        for &y0 in &by {
            for &x0 in &bx {
                let rect = |ch: usize| {
                    scflat[idx(y0 + bs, x0 + bs, ch)]
                        - scflat[idx(y0, x0 + bs, ch)]
                        - scflat[idx(y0 + bs, x0, ch)]
                        + scflat[idx(y0, x0, ch)]
                };
                let mut msq = 0f64;
                for ch in 0..c {
                    let m = rect(ch) / area;
                    msq += m * m;
                }
                let bvar = rect(c) / area - msq;
                let cx = x0 as f64 + bs as f64 / 2.0;
                let cy = y0 as f64 + bs as f64 / 2.0;
                all_px.push(cx);
                all_py.push(cy);
                if bvar > thresh {
                    act_px.push(cx);
                    act_py.push(cy);
                    active_sum += bvar;
                    active_cnt += 1;
                }
            }
        }
        let (mut px, mut py) = if act_px.len() < 8 {
            (all_px, all_py)
        } else {
            (act_px, act_py)
        };
        let max_points = 2600usize;
        if px.len() > max_points {
            let m = px.len();
            let sel: Vec<usize> = (0..max_points)
                .map(|i| (i as f64 * (m as f64 - 1.0) / (max_points as f64 - 1.0)) as usize)
                .collect();
            let mut pairs: Vec<(f64, f64)> = sel.iter().map(|&i| (px[i], py[i])).collect();
            pairs.sort_by(|a, b| {
                (a.1 * w as f64 + a.0)
                    .partial_cmp(&(b.1 * w as f64 + b.0))
                    .unwrap()
            });
            px = pairs.iter().map(|p| p.0).collect();
            py = pairs.iter().map(|p| p.1).collect();
        }
        let active_var = if active_cnt > 0 {
            active_sum / active_cnt as f64
        } else {
            total_var
        };
        CellVarContrast {
            w,
            h,
            c,
            scflat,
            total_var,
            active_var,
            px,
            py,
        }
    }

    /// Mean within-cell variance over active points (integer-snapped
    /// corners, fused-SAT gathers). Mirrors _cells_variance + .mean().
    fn cells_variance_mean(&self, step: f64, phase_x: f64, phase_y: f64) -> f64 {
        let stride = self.w + 1;
        let cw = self.c + 1;
        let mut acc = 0f64;
        for i in 0..self.px.len() {
            let x0 = phase_x + ((self.px[i] - phase_x) / step).floor() * step;
            let y0 = phase_y + ((self.py[i] - phase_y) / step).floor() * step;
            let ix0 = pyround(x0).clamp(0.0, self.w as f64 - 1.0) as usize;
            let iy0 = pyround(y0).clamp(0.0, self.h as f64 - 1.0) as usize;
            let ix1 = pyround(x0 + step).clamp(ix0 as f64 + 1.0, self.w as f64) as usize;
            let iy1 = pyround(y0 + step).clamp(iy0 as f64 + 1.0, self.h as f64) as usize;
            let g = |y: usize, x: usize, ch: usize| self.scflat[(y * stride + x) * cw + ch];
            let area = ((ix1 - ix0) * (iy1 - iy0)) as f64;
            let mut msq = 0f64;
            for ch in 0..self.c {
                let s1 = g(iy1, ix1, ch) - g(iy1, ix0, ch) - g(iy0, ix1, ch) + g(iy0, ix0, ch);
                let m = s1 / area;
                msq += m * m;
            }
            let s2 = g(iy1, ix1, self.c) - g(iy1, ix0, self.c) - g(iy0, ix1, self.c)
                + g(iy0, ix0, self.c);
            acc += (s2 / area - msq).max(0.0);
        }
        acc / self.px.len().max(1) as f64
    }

    /// Global-phase contrast (varcontrast.contrast, n_phases=3).
    fn contrast(&self, step: f64, n_phases: usize) -> f64 {
        if step < 1.5
            || step > self.w as f64 / 3.0
            || step > self.h as f64 / 3.0
            || self.total_var <= 1e-9
            || self.px.is_empty()
        {
            return 0.0;
        }
        let mut best = f64::INFINITY;
        let mut worst = f64::MIN;
        for iy in 0..n_phases {
            let py = iy as f64 * (step / n_phases as f64);
            for ix in 0..n_phases {
                let px = ix as f64 * (step / n_phases as f64);
                let v = self.cells_variance_mean(step, px, py);
                if v < best {
                    best = v;
                }
                if v > worst {
                    worst = v;
                }
            }
        }
        (worst - best) / (best + 0.05 * self.active_var)
    }

    /// (log-steps, z) detrended prominence curve (scored_curve).
    pub fn z_curve(&self) -> (Vec<f64>, Vec<f64>) {
        use rayon::prelude::*;
        let l = self.w.min(self.h) as f64;
        let max_step = (l / 8.0).max(4.0).min(64.0);
        let mut steps = Vec::new();
        let mut s = 2.0f64;
        while s <= max_step {
            steps.push(s);
            s *= 1.04;
        }
        let cs: Vec<f64> = steps.par_iter().map(|&s| self.contrast(s, 3)).collect();
        let base = median_filter1d(&cs, 15);
        let resid: Vec<f64> = cs.iter().zip(base.iter()).map(|(&c, &b)| c - b).collect();
        let absr: Vec<f64> = resid.iter().map(|&r| r.abs()).collect();
        let sigma = median(&absr) * 1.4826 + 1e-9;
        let z: Vec<f64> = resid.iter().map(|&r| r / sigma).collect();
        let logs: Vec<f64> = steps.iter().map(|&v| v.ln()).collect();
        // store step range endpoints inside logs for the z_of gate
        (logs, z)
    }
}

/// z_of(step) closure data: (log_steps, z) from z_curve; gate outside range.
pub fn vc_z_of(logs: &[f64], z: &[f64], step: f64) -> f64 {
    if logs.is_empty() {
        return 0.0;
    }
    let ls = step.ln();
    if ls < logs[0] || ls > logs[logs.len() - 1] {
        return 0.0;
    }
    crate::sigproc::interp(ls, logs, z)
}
