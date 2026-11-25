//! Reconstruction: detected grid -> native-resolution pixel art.
//! Mirrors detector/reconstruct.py (color="mode" path; palette_snap is not
//! ported - the server API default is palette_snap=False).

use rayon::prelude::*;

/// python/np.round banker's rounding
fn pyround(v: f64) -> f64 {
    v.round_ties_even()
}

/// |d1| of luminance (alpha-weighted) summed across the perpendicular axis.
/// axis=0: diff along x, sum over rows -> length w-1 (values f32-rounded).
fn axis_profile(rgba: &[u8], w: usize, h: usize, axis: usize) -> Vec<f64> {
    let mut g = vec![0f32; w * h];
    for i in 0..w * h {
        let p = i * 4;
        let a = rgba[p + 3] as f32 / 255.0;
        g[i] =
            (0.299 * rgba[p] as f32 + 0.587 * rgba[p + 1] as f32 + 0.114 * rgba[p + 2] as f32) * a;
    }
    if axis == 0 {
        // diff along x, sum over rows (f64 accumulate, f32-rounded like numpy)
        let mut acc = vec![0f64; w - 1];
        for y in 0..h {
            let r = y * w;
            for x in 0..w - 1 {
                acc[x] += (g[r + x + 1] - g[r + x]).abs() as f64;
            }
        }
        acc.iter().map(|&v| v as f32 as f64).collect()
    } else {
        let mut acc = vec![0f64; h - 1];
        for y in 0..h - 1 {
            let r0 = y * w;
            let r1 = (y + 1) * w;
            let mut s = 0f64;
            for x in 0..w {
                s += (g[r1 + x] - g[r0 + x]).abs() as f64;
            }
            acc[y] = s as f32 as f64;
        }
        acc
    }
}

/// Integral images for within-cell variance of arbitrary cut sets.
struct Sat {
    s1: Vec<[f64; 3]>, // (h+1)*(w+1)
    s2: Vec<f64>,
    w: usize,
    h: usize,
}

impl Sat {
    fn new(rgba: &[u8], w: usize, h: usize) -> Sat {
        let stride = w + 1;
        let mut s1 = vec![[0f64; 3]; (h + 1) * stride];
        let mut s2 = vec![0f64; (h + 1) * stride];
        for y in 0..h {
            let mut row1 = [0f64; 3];
            let mut row2 = 0f64;
            for x in 0..w {
                let p = (y * w + x) * 4;
                let r = rgba[p] as f64;
                let g = rgba[p + 1] as f64;
                let b = rgba[p + 2] as f64;
                row1[0] += r;
                row1[1] += g;
                row1[2] += b;
                row2 += r * r + g * g + b * b;
                let up = (y) * stride + (x + 1);
                let cur = (y + 1) * stride + (x + 1);
                s1[cur] = [
                    s1[up][0] + row1[0],
                    s1[up][1] + row1[1],
                    s1[up][2] + row1[2],
                ];
                s2[cur] = s2[up] + row2;
            }
        }
        Sat { s1, s2, w, h }
    }

    fn cell_var(&self, xs: &[i64], ys: &[i64]) -> f64 {
        let stride = self.w + 1;
        let mut num = 0f64;
        let mut den = 0f64;
        for yi in 0..ys.len() - 1 {
            let (y0, y1) = (ys[yi] as usize, ys[yi + 1] as usize);
            for xi in 0..xs.len() - 1 {
                let (x0, x1) = (xs[xi] as usize, xs[xi + 1] as usize);
                let idx = |y: usize, x: usize| y * stride + x;
                let a = idx(y0, x0);
                let b = idx(y0, x1);
                let c = idx(y1, x0);
                let d = idx(y1, x1);
                let s2 = self.s2[d] - self.s2[b] - self.s2[c] + self.s2[a];
                let area = (((y1 - y0) * (x1 - x0)) as f64).max(1.0);
                let mut m2 = 0f64;
                for ch in 0..3 {
                    let s1 = self.s1[d][ch] - self.s1[b][ch] - self.s1[c][ch] + self.s1[a][ch];
                    let m = s1 / area;
                    m2 += m * m;
                }
                let v = s2 / area - m2;
                num += v * area;
                den += area;
            }
        }
        num / den.max(1e-12)
    }
}

/// unique(clip(round(arange(phase, extent+step, step)),0,extent)) with 0 front
fn phase_cuts(phase: f64, extent: usize, step: f64) -> Vec<i64> {
    let stop = extent as f64 + step;
    let n = ((stop - phase) / step).ceil().max(0.0) as usize;
    let mut v: Vec<i64> = (0..n)
        .map(|i| pyround(phase + i as f64 * step).clamp(0.0, extent as f64) as i64)
        .collect();
    v.sort_unstable();
    v.dedup();
    if v.first() != Some(&0) {
        v.insert(0, 0);
    }
    v
}

/// Grid phase minimizing within-cell variance (coarse 5x5 SAT sweep).
fn best_phase(sat: &Sat, step_x: f64, step_y: f64) -> (f64, f64) {
    let n = 5;
    let mut best = (0f64, 0f64, f64::INFINITY);
    for iy in 0..n {
        let py = iy as f64 * (step_y / n as f64);
        let ys = phase_cuts(py, sat.h, step_y);
        for ix in 0..n {
            let px = ix as f64 * (step_x / n as f64);
            let xs = phase_cuts(px, sat.w, step_x);
            let v = sat.cell_var(&xs, &ys);
            if v < best.2 {
                best = (px, py, v);
            }
        }
    }
    (best.0, best.1)
}

/// (phase, strength): sub-pixel comb phase of the cut-energy profile.
fn comb_phase(profile: &[f64], step: f64) -> (f64, f64) {
    let n = profile.len();
    if step < 1.5 || (n as f64) < 3.0 * step {
        return (0.0, 0.0);
    }
    let n_ph = std::cmp::max(8, pyround(step * 4.0) as usize);
    let kmax = ((n as f64 - 2.0) / step) as usize; // ks = 0..=kmax
    let mut resp = vec![0f64; n_ph];
    for b in 0..n_ph {
        let phase = b as f64 * (step / n_ph as f64);
        let mut acc = 0f64;
        let mut cnt = 0usize;
        for k in 0..=kmax {
            let pos = phase + k as f64 * step;
            if pos <= n as f64 - 2.0 {
                let i0 = (pos as i64).clamp(0, n as i64 - 2) as usize;
                let f = pos - i0 as f64;
                acc += profile[i0] * (1.0 - f) + profile[i0 + 1] * f;
                cnt += 1;
            }
        }
        resp[b] = acc / cnt.max(1) as f64;
    }
    let mut b = 0;
    for i in 0..n_ph {
        if resp[i] > resp[b] {
            b = i;
        }
    }
    let mean = resp.iter().sum::<f64>() / n_ph as f64;
    let strength = resp[b] / (mean + 1e-9);
    let lo = resp[(b + n_ph - 1) % n_ph];
    let hi = resp[(b + 1) % n_ph];
    let denom = lo - 2.0 * resp[b] + hi;
    let d = if denom.abs() > 1e-12 {
        (0.5 * (lo - hi) / denom).clamp(-1.0, 1.0)
    } else {
        0.0
    };
    let ph = (b as f64 * (step / n_ph as f64) + d * (step / n_ph as f64)).rem_euclid(step);
    (ph, strength)
}

/// Exactly n_cells+1 cuts: lattice targets phase+k*step, interior cuts
/// snapped to the local |d1| max, sliver-guarded (min gap 55% of step).
fn snapped_cuts(profile: &[f64], step: f64, phase: f64, extent: usize, n_cells: usize) -> Vec<i64> {
    let mut cuts = vec![0i64; n_cells + 1];
    cuts[n_cells] = extent as i64;
    let rad = std::cmp::max(1, pyround(step * 0.30) as i64);
    let prof_mean = if profile.is_empty() {
        0.0
    } else {
        profile.iter().sum::<f64>() / profile.len() as f64
    };
    let min_gap = std::cmp::max(1, (0.55 * step).floor() as i64);
    let first = phase.rem_euclid(step);
    let base = if first >= 0.5 * step {
        first
    } else {
        first + step
    };
    let mut prev = 0i64;
    for j in 1..n_cells {
        let t = base + (j as f64 - 1.0) * step;
        let mut c = pyround(t.max(1.0).min(extent as f64 - 1.0)) as i64;
        let lo = std::cmp::max(prev + min_gap, c - rad);
        let hi = *[
            extent as i64 - 1,
            c + rad,
            extent as i64 - (n_cells - j) as i64 * min_gap,
        ]
        .iter()
        .min()
        .unwrap();
        if hi < lo {
            c = std::cmp::min(prev + min_gap, extent as i64 - (n_cells - j) as i64);
        } else {
            let seg = &profile[lo as usize..=(hi as usize)];
            let smax = seg.iter().cloned().fold(f64::MIN, f64::max);
            if !seg.is_empty() && smax >= 0.5 * prof_mean && smax > 1e-9 {
                let mut am = 0;
                for i in 0..seg.len() {
                    if seg[i] > seg[am] {
                        am = i;
                    }
                }
                c = lo + am as i64;
            } else {
                c = c.clamp(lo, hi);
            }
        }
        c = std::cmp::max(
            std::cmp::min(c, extent as i64 - (n_cells - j) as i64),
            prev + 1,
        );
        cuts[j] = c;
        prev = c;
    }
    cuts
}

/// Bend global cut lines into per-band polylines.
/// Returns (band_edges, cuts[n_bands][n_cuts] as f64).
fn banded_cuts(
    rgba: &[u8],
    w: usize,
    h: usize,
    base_cuts: &[i64],
    step: f64,
    axis: usize,
) -> (Vec<usize>, Vec<Vec<f64>>) {
    let perp = if axis == 0 { h } else { w };
    let extent = if axis == 0 { w } else { h };
    let n_bands = ((perp as f64 / (step * 7.0)).clamp(1.0, 12.0)) as usize;
    let bstep = perp as f64 / n_bands as f64;
    let edges: Vec<usize> = (0..=n_bands).map(|i| (i as f64 * bstep) as usize).collect();
    if n_bands <= 1 {
        return (edges, vec![base_cuts.iter().map(|&c| c as f64).collect()]);
    }

    let mut g = vec![0f32; w * h];
    for i in 0..w * h {
        let p = i * 4;
        let a = rgba[p + 3] as f32 / 255.0;
        g[i] =
            (0.299 * rgba[p] as f32 + 0.587 * rgba[p + 1] as f32 + 0.114 * rgba[p + 2] as f32) * a;
    }

    let rad = std::cmp::max(1, pyround(step * 0.3) as i64);
    let min_gap = std::cmp::max(1, (0.55 * step).floor() as i64) as f64;
    let n_cuts = base_cuts.len();
    let mut cuts: Vec<Vec<f64>> = (0..n_bands)
        .map(|_| base_cuts.iter().map(|&c| c as f64).collect())
        .collect();

    let profs: Vec<Vec<f64>> = (0..n_bands)
        .into_par_iter()
        .map(|b| {
            // prof[c] = |d1| energy of a cut between positions c-1 and c
            let mut prof = vec![0f64; extent];
            if axis == 0 {
                for y in edges[b]..edges[b + 1] {
                    let r = y * w;
                    for x in 0..w - 1 {
                        prof[x + 1] += (g[r + x + 1] - g[r + x]).abs() as f64;
                    }
                }
            } else {
                for y in 0..h - 1 {
                    let r0 = y * w;
                    let r1 = (y + 1) * w;
                    let mut s = 0f64;
                    for x in edges[b]..edges[b + 1] {
                        s += (g[r1 + x] - g[r0 + x]).abs() as f64;
                    }
                    prof[y + 1] = s as f32 as f64;
                }
            }
            if axis == 0 {
                // f32-round the row-summed profile like numpy
                for v in prof.iter_mut() {
                    *v = *v as f32 as f64;
                }
            }
            prof
        })
        .collect();

    for b in 0..n_bands {
        let prof = &profs[b];
        let pmean = prof.iter().sum::<f64>() / prof.len() as f64;
        for j in 1..n_cuts - 1 {
            let c = base_cuts[j];
            let lo = std::cmp::max(cuts[b][j - 1] as i64 + min_gap as i64, c - rad);
            let hi = *[extent as i64 - 1, c + rad, base_cuts[j + 1] - 1]
                .iter()
                .min()
                .unwrap();
            if hi <= lo {
                cuts[b][j] = (cuts[b][j - 1] + min_gap).max((c as f64).min(extent as f64 - 1.0));
                continue;
            }
            let sl = &prof[lo as usize..=(hi as usize)];
            let smax = sl.iter().cloned().fold(f64::MIN, f64::max);
            if !sl.is_empty() && smax >= 0.5 * pmean && smax > 1e-9 {
                let mut am = 0;
                for i in 0..sl.len() {
                    if sl[i] > sl[am] {
                        am = i;
                    }
                }
                cuts[b][j] = (lo + am as i64) as f64;
            } else {
                cuts[b][j] = (c as f64).clamp(lo as f64, hi as f64);
            }
        }
    }
    if n_bands >= 3 {
        let orig = cuts.clone();
        for b in 1..n_bands - 1 {
            for j in 0..n_cuts {
                cuts[b][j] = 0.5 * orig[b][j] + 0.25 * (orig[b - 1][j] + orig[b + 1][j]);
            }
        }
    }
    for b in 0..n_bands {
        for j in 1..n_cuts {
            if cuts[b][j] < cuts[b][j - 1] {
                cuts[b][j] = cuts[b][j - 1];
            }
        }
        cuts[b][0] = 0.0;
        cuts[b][n_cuts - 1] = extent as f64;
    }
    (edges, cuts)
}

/// searchsorted(a, v, side="right") - 1
fn cell_of(a: &[f64], v: f64) -> i64 {
    let mut lo = 0usize;
    let mut hi = a.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if a[mid] <= v {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo as i64 - 1
}

/// Per-pixel warped cell index along one axis from banded cut polylines.
/// Returns (perp_len x extent) row-major.
fn warped_index(
    bands: &[Vec<f64>],
    edges: &[usize],
    perp_len: usize,
    extent: usize,
    n_cells: usize,
) -> Vec<i64> {
    let n_b = bands.len();
    let n_cuts = bands[0].len();
    let mut out = vec![0i64; perp_len * extent];
    if n_b == 1 {
        let row: Vec<i64> = (0..extent)
            .map(|x| cell_of(&bands[0], x as f64 + 0.5).clamp(0, n_cells as i64 - 1))
            .collect();
        for r in 0..perp_len {
            out[r * extent..(r + 1) * extent].copy_from_slice(&row);
        }
        return out;
    }
    let centers: Vec<f64> = (0..n_b)
        .map(|i| (edges[i] + edges[i + 1]) as f64 / 2.0)
        .collect();
    out.par_chunks_mut(extent)
        .enumerate()
        .for_each(|(r, orow)| {
            let t = r as f64;
            // continuous polyline positions at this scanline (monotone-forced)
            let mut line = vec![0f64; n_cuts];
            // np.interp over band centers with clamped extrapolation
            let seg = if t <= centers[0] {
                0
            } else if t >= centers[n_b - 1] {
                n_b - 1
            } else {
                let mut s = 0;
                while s + 1 < n_b && centers[s + 1] < t {
                    s += 1;
                }
                s
            };
            for j in 0..n_cuts {
                let v = if t <= centers[0] {
                    bands[0][j]
                } else if t >= centers[n_b - 1] {
                    bands[n_b - 1][j]
                } else {
                    let f = (t - centers[seg]) / (centers[seg + 1] - centers[seg]);
                    bands[seg][j] * (1.0 - f) + bands[seg + 1][j] * f
                };
                line[j] = if j > 0 && v < line[j - 1] {
                    line[j - 1]
                } else {
                    v
                };
            }
            for x in 0..extent {
                orow[x] = cell_of(&line, x as f64 + 0.5).clamp(0, n_cells as i64 - 1);
            }
        });
    out
}

pub struct ReconOut {
    pub rgba: Vec<u8>, // ncx * ncy * 4
    pub cols: usize,
    pub rows: usize,
}

pub fn reconstruct(
    rgba: &[u8],
    w: usize,
    h: usize,
    step_x: f64,
    step_y: f64,
    cols: usize,
    rows: usize,
    dark_stroke: bool,
    auto_palette: bool,
) -> ReconOut {
    let sat = Sat::new(rgba, w, h);

    // profile[c] = |d1| energy of a cut between columns c-1 and c
    let mut prof_x = vec![0f64];
    prof_x.extend(axis_profile(rgba, w, h, 0));
    let mut prof_y = vec![0f64];
    prof_y.extend(axis_profile(rgba, w, h, 1));

    let (mut px, sx_str) = comb_phase(&prof_x, step_x);
    let (mut py, sy_str) = comb_phase(&prof_y, step_y);
    if sx_str.min(sy_str) < 1.35 {
        let bp = best_phase(&sat, step_x, step_y);
        px = bp.0;
        py = bp.1;
    }

    // candidate cut sets: snapped / phase lattice / phase-0 lattice
    let flat_x = vec![0f64; prof_x.len()];
    let flat_y = vec![0f64; prof_y.len()];
    let cand_x = [
        snapped_cuts(&prof_x, step_x, px, w, cols),
        snapped_cuts(&flat_x, step_x, px, w, cols),
        snapped_cuts(&flat_x, step_x, 0.0, w, cols),
    ];
    let cand_y = [
        snapped_cuts(&prof_y, step_y, py, h, rows),
        snapped_cuts(&flat_y, step_y, py, h, rows),
        snapped_cuts(&flat_y, step_y, 0.0, h, rows),
    ];
    let mut xs = &cand_x[0];
    {
        let mut best = f64::INFINITY;
        for c in cand_x.iter() {
            let v = sat.cell_var(c, &cand_y[2]);
            if v < best {
                best = v;
                xs = c;
            }
        }
    }
    let mut ys = &cand_y[0];
    {
        let mut best = f64::INFINITY;
        for c in cand_y.iter() {
            let v = sat.cell_var(xs, c);
            if v < best {
                best = v;
                ys = c;
            }
        }
    }
    let xs = xs.clone();
    let ys = ys.clone();

    let ncx = xs.len() - 1;
    let ncy = ys.len() - 1;
    let n = ncx * ncy;
    let xsf: Vec<f64> = xs.iter().map(|&v| v as f64).collect();
    let ysf: Vec<f64> = ys.iter().map(|&v| v as f64).collect();
    let ix: Vec<i64> = (0..w)
        .map(|x| cell_of(&xsf, x as f64).clamp(0, ncx as i64 - 1))
        .collect();
    let iy: Vec<i64> = (0..h)
        .map(|y| cell_of(&ysf, y as f64).clamp(0, ncy as i64 - 1))
        .collect();
    let mut cell: Vec<i64> = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            cell.push(iy[y] * ncx as i64 + ix[x]);
        }
    }

    // rgb in [0,1]
    let rgb: Vec<[f64; 3]> = (0..w * h)
        .map(|i| {
            let p = i * 4;
            [
                rgba[p] as f64 / 255.0,
                rgba[p + 1] as f64 / 255.0,
                rgba[p + 2] as f64 / 255.0,
            ]
        })
        .collect();

    let pooled_var = |cell_idx: &[i64]| -> f64 {
        let mut c1 = vec![0f64; n];
        let mut m = vec![[0f64; 3]; n];
        let mut q = vec![[0f64; 3]; n];
        for (i, &ci) in cell_idx.iter().enumerate() {
            let ci = ci as usize;
            c1[ci] += 1.0;
            for ch in 0..3 {
                m[ci][ch] += rgb[i][ch];
                q[ci][ch] += rgb[i][ch] * rgb[i][ch];
            }
        }
        let mut num = 0f64;
        let mut den = 0f64;
        for ci in 0..n {
            let c = c1[ci].max(1.0);
            let mut v = 0f64;
            for ch in 0..3 {
                let mm = m[ci][ch] / c;
                let qq = q[ci][ch] / c;
                v += (qq - mm * mm).max(0.0);
            }
            num += v * c;
            den += c;
        }
        num / den.max(1e-12)
    };

    // banded (warped) refinement, adopted only when variance drops >= 0.5%
    {
        let ((x_edges, xs_b), (y_edges, ys_b)) = rayon::join(
            || banded_cuts(rgba, w, h, &xs, step_x, 0),
            || banded_cuts(rgba, w, h, &ys, step_y, 1),
        );
        if xs_b.len() > 1 || ys_b.len() > 1 {
            let (ixw, iyw) = rayon::join(
                || warped_index(&xs_b, &x_edges, h, w, ncx),
                || warped_index(&ys_b, &y_edges, w, h, ncy),
            );
            // iyw is (w, h) row-major; transpose access
            let mut cell_w: Vec<i64> = Vec::with_capacity(w * h);
            for y in 0..h {
                for x in 0..w {
                    cell_w.push(iyw[x * h + y] * ncx as i64 + ixw[y * w + x]);
                }
            }
            if pooled_var(&cell_w) < 0.995 * pooled_var(&cell) {
                cell = cell_w;
            }
        }
    }

    // center weights: triangular within each straight-cut cell span
    let mut wx = vec![0f64; w];
    for x in 0..w {
        let i = ix[x] as usize;
        let lo = xs[i] as f64;
        let hi = xs[i + 1] as f64;
        let f = (x as f64 + 0.5 - lo) / (hi - lo).max(1.0);
        wx[x] = 1.0 - 2.0 * (f - 0.5).abs();
    }
    let mut wy = vec![0f64; h];
    for y in 0..h {
        let i = iy[y] as usize;
        let lo = ys[i] as f64;
        let hi = ys[i + 1] as f64;
        let f = (y as f64 + 0.5 - lo) / (hi - lo).max(1.0);
        wy[y] = 1.0 - 2.0 * (f - 0.5).abs();
    }
    let wgt: Vec<f64> = (0..w * h).map(|i| wy[i / w] * wx[i % w] + 1e-4).collect();

    let mut cnt = vec![0f64; n];
    let mut mean = vec![[0f64; 3]; n];
    for (i, &ci) in cell.iter().enumerate() {
        let ci = ci as usize;
        cnt[ci] += 1.0;
        for ch in 0..3 {
            mean[ci][ch] += rgb[i][ch];
        }
    }
    let cnt: Vec<f64> = cnt.iter().map(|&c| c.max(1.0)).collect();
    for ci in 0..n {
        for ch in 0..3 {
            mean[ci][ch] /= cnt[ci];
        }
    }

    // ---- 5-bit binned local mode with center-weighted votes
    let key: Vec<u32> = (0..w * h)
        .map(|i| {
            let p = i * 4;
            (((rgba[p] as u32) >> 3) << 10)
                | (((rgba[p + 1] as u32) >> 3) << 5)
                | ((rgba[p + 2] as u32) >> 3)
        })
        .collect();
    // global per-bin mean colors
    let mut gcnt = vec![0f64; 32768];
    let mut gmean = vec![[0f64; 3]; 32768];
    for i in 0..w * h {
        let k = key[i] as usize;
        gcnt[k] += 1.0;
        for ch in 0..3 {
            gmean[k][ch] += rgb[i][ch];
        }
    }
    for k in 0..32768 {
        let c = gcnt[k].max(1.0);
        for ch in 0..3 {
            gmean[k][ch] /= c;
        }
    }

    // per-cell winning bin: max weighted vote, ties -> larger bin key
    let binned_mode = |sel: &dyn Fn(usize) -> bool| -> Vec<Option<u32>> {
        use std::collections::HashMap;
        let mut votes: HashMap<u64, f64> = HashMap::new();
        for i in 0..w * h {
            if sel(i) {
                let comp = (cell[i] as u64) * 32768 + key[i] as u64;
                *votes.entry(comp).or_insert(0.0) += wgt[i];
            }
        }
        let mut win: Vec<Option<(f64, u32)>> = vec![None; n];
        let mut comps: Vec<(&u64, &f64)> = votes.iter().collect();
        comps.sort_by_key(|(c, _)| **c); // ascending comp = lexsort tiebreak
        for (&comp, &v) in comps {
            let ci = (comp / 32768) as usize;
            let k = (comp % 32768) as u32;
            match win[ci] {
                Some((bv, _)) if v < bv => {}
                Some((bv, _)) if v == bv => win[ci] = Some((v, k)), // last wins
                _ if win[ci].is_some() && win[ci].unwrap().0 > v => {}
                _ => win[ci] = Some((v, k)),
            }
        }
        win.iter().map(|o| o.map(|(_, k)| k)).collect()
    };

    let all_win = binned_mode(&|_| true);
    let mut out: Vec<[f64; 3]> = (0..n)
        .map(|ci| match all_win[ci] {
            Some(k) => gmean[k as usize],
            None => mean[ci],
        })
        .collect();

    if dark_stroke {
        let t_thr = 38.0 / 255.0;
        let luma_px: Vec<f64> = rgb
            .iter()
            .map(|c| 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2])
            .collect();
        let mut min_luma = vec![2.0f64; n];
        for i in 0..w * h {
            let ci = cell[i] as usize;
            if luma_px[i] < min_luma[ci] {
                min_luma[ci] = luma_px[i];
            }
        }
        let dark_px: Vec<bool> = (0..w * h)
            .map(|i| {
                let ci = cell[i] as usize;
                luma_px[i] <= min_luma[ci] + (8.0 / 255.0f64).max(0.35 * t_thr)
            })
            .collect();
        let mut dark_cnt = vec![0f64; n];
        for i in 0..w * h {
            if dark_px[i] {
                dark_cnt[cell[i] as usize] += 1.0;
            }
        }
        let need: Vec<bool> = (0..n)
            .map(|ci| {
                let out_luma = 0.299 * out[ci][0] + 0.587 * out[ci][1] + 0.114 * out[ci][2];
                out_luma - min_luma[ci] >= t_thr
                    && dark_cnt[ci] >= (2.0f64).max(0.08 * cnt[ci])
                    && dark_cnt[ci] <= (0.42 * cnt[ci]).ceil()
            })
            .collect();
        let any_sel = (0..w * h).any(|i| need[cell[i] as usize] && dark_px[i]);
        if any_sel {
            let dark_win = binned_mode(&|i| need[cell[i] as usize] && dark_px[i]);
            for ci in 0..n {
                if need[ci] {
                    if let Some(k) = dark_win[ci] {
                        out[ci] = gmean[k as usize];
                    }
                }
            }
        }
    }

    // OPTIMAL PALETTE: snap the (structurally correct but color-muddy) mode
    // output onto a small Wu palette. K is detected on this already-denoised
    // 1x output (reliable), so soft AA boundary cells collapse onto real
    // pixel-art colors: crisp detail + a flat palette. See detector/wu.py.
    if auto_palette {
        let wpx: Vec<[u8; 3]> = (0..n)
            .map(|ci| {
                [
                    pyround(out[ci][0] * 255.0).clamp(0.0, 255.0) as u8,
                    pyround(out[ci][1] * 255.0).clamp(0.0, 255.0) as u8,
                    pyround(out[ci][2] * 255.0).clamp(0.0, 255.0) as u8,
                ]
            })
            .collect();
        let k = crate::wu::elbow_color_count(&wpx, 64);
        let (pal, labels) = crate::wu::quantize(&wpx, k);
        for ci in 0..n {
            out[ci] = pal[labels[ci]];
        }
    }

    // alpha: majority of pixels with a > 127
    let mut asum = vec![0f64; n];
    for i in 0..w * h {
        if rgba[i * 4 + 3] > 127 {
            asum[cell[i] as usize] += 1.0;
        }
    }

    let mut out_rgba = vec![0u8; n * 4];
    for ci in 0..n {
        for ch in 0..3 {
            // np.rint = banker's rounding
            out_rgba[ci * 4 + ch] = pyround(out[ci][ch] * 255.0).clamp(0.0, 255.0) as u8;
        }
        out_rgba[ci * 4 + 3] = if asum[ci] / cnt[ci] > 0.5 { 255 } else { 0 };
    }
    ReconOut {
        rgba: out_rgba,
        cols: ncx,
        rows: ncy,
    }
}

/// Two-stage packing on a regular even grid (mirrors
/// detector.reconstruct.two_stage_pack).
///
/// Stage 1 (STRUCTURE): quantise to a small palette (adaptive K) and let each
/// cell vote among the clean quantised labels -> crisp placement.
/// Stage 2 (COLOUR): colour each cell from the ORIGINAL pixels carrying the
/// winning label -> crisp lines AND accurate, un-clamped colours.
pub fn two_stage_pack(
    rgba: &[u8],
    w: usize,
    h: usize,
    cols: usize,
    rows: usize,
    k_colors: usize,
) -> ReconOut {
    let k_req = if k_colors > 0 {
        k_colors
    } else {
        crate::kmeans::adaptive_k(rgba, w, h, 16, 48, 0.003)
    };
    let (labels, kc) = crate::kmeans::kmeans_labels(rgba, w, h, k_req);
    let kc = kc.max(1);
    let n = cols * rows;
    let cw = w as f64 / cols as f64;
    let ch = h as f64 / rows as f64;

    let mut ixs = vec![0usize; w];
    let mut wxs = vec![0f64; w];
    for x in 0..w {
        let ix = ((x * cols) / w).min(cols - 1);
        ixs[x] = ix;
        let fx = (x as f64 + 0.5 - ix as f64 * cw) / cw;
        wxs[x] = 1.0 - 2.0 * (fx - 0.5).abs();
    }
    let mut iys = vec![0usize; h];
    let mut wys = vec![0f64; h];
    for y in 0..h {
        let iy = ((y * rows) / h).min(rows - 1);
        iys[y] = iy;
        let fy = (y as f64 + 0.5 - iy as f64 * ch) / ch;
        wys[y] = 1.0 - 2.0 * (fy - 0.5).abs();
    }

    // stage 1: winning label per cell (centre-weighted vote over labels)
    let mut wsum = vec![0f64; n * kc];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let cell = iys[y] * cols + ixs[x];
            let wgt = wys[y] * wxs[x] + 1e-4;
            wsum[cell * kc + labels[i] as usize] += wgt;
        }
    }
    let mut win = vec![0u32; n];
    for c in 0..n {
        let base = c * kc;
        let mut bi = 0usize;
        let mut bv = wsum[base];
        for l in 1..kc {
            if wsum[base + l] > bv {
                bv = wsum[base + l];
                bi = l;
            }
        }
        win[c] = bi as u32;
    }

    // stage 2: colour from original pixels carrying the winning label
    let mut csum = vec![[0f64; 3]; n];
    let mut wden = vec![0f64; n];
    let mut selcnt = vec![0f64; n];
    let mut cnt = vec![0f64; n];
    let mut msum = vec![[0f64; 3]; n];
    let mut asum = vec![0f64; n];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let p = i * 4;
            let cell = iys[y] * cols + ixs[x];
            let wgt = wys[y] * wxs[x] + 1e-4;
            let rgb = [
                rgba[p] as f64 / 255.0,
                rgba[p + 1] as f64 / 255.0,
                rgba[p + 2] as f64 / 255.0,
            ];
            cnt[cell] += 1.0;
            for c in 0..3 {
                msum[cell][c] += rgb[c];
            }
            if rgba[p + 3] > 127 {
                asum[cell] += 1.0;
            }
            if labels[i] == win[cell] {
                selcnt[cell] += 1.0;
                wden[cell] += wgt;
                for c in 0..3 {
                    csum[cell][c] += rgb[c] * wgt;
                }
            }
        }
    }

    let mut out_rgba = vec![0u8; n * 4];
    for c in 0..n {
        let color = if selcnt[c] >= 0.5 && wden[c] > 1e-9 {
            [
                csum[c][0] / wden[c],
                csum[c][1] / wden[c],
                csum[c][2] / wden[c],
            ]
        } else {
            let cc = cnt[c].max(1.0);
            [msum[c][0] / cc, msum[c][1] / cc, msum[c][2] / cc]
        };
        for ch in 0..3 {
            out_rgba[c * 4 + ch] = pyround(color[ch] * 255.0).clamp(0.0, 255.0) as u8;
        }
        out_rgba[c * 4 + 3] = if asum[c] / cnt[c].max(1.0) > 0.5 {
            255
        } else {
            0
        };
    }
    ReconOut {
        rgba: out_rgba,
        cols,
        rows,
    }
}
