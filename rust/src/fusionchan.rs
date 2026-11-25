//! Lean fusion evidence + the four ACTIVE channels the full-mode arbiter
//! reads: ray_e1, tile_e1, tile_e2, spec_e1. Mirrors detector/fusion.py
//! build_evidence(lean=True) + channel_matrix(only=ACTIVE) + fused_curve,
//! with the supporting pieces from detector/channels.py.

use crate::kmeans::kmeans_quantize_rgba;
use crate::sigproc::*;
use rayon::prelude::*;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

/// fusion.py ladder(): geometric steps 2.0 .. 64.0, ratio 1.03,
/// each value rounded to 4 decimals (running product unrounded).
pub fn ladder() -> Vec<f64> {
    let mut steps = Vec::new();
    let mut s = 2.0f64;
    while s <= 64.0 + 1e-9 {
        steps.push((s * 1e4).round_ties_even() / 1e4);
        s *= 1.03;
    }
    steps
}

/// float32 (H, W, C) with alpha premultiplied and kept as a channel.
/// Returns flat channel-major-last data (h * w * 4).
fn flatten_channels(rgba: &[u8], w: usize, h: usize) -> Vec<f32> {
    let mut img = vec![0f32; w * h * 4];
    for i in 0..w * h {
        let p = i * 4;
        let a = rgba[p + 3] as f32 / 255.0;
        img[i * 4] = rgba[p] as f32 * a;
        img[i * 4 + 1] = rgba[p + 1] as f32 * a;
        img[i * 4 + 2] = rgba[p + 2] as f32 * a;
        img[i * 4 + 3] = rgba[p + 3] as f32;
    }
    img
}

/// E1 cut-energy profiles for both axes: e1x (W+1), e1y (H+1); borders set
/// to the top interior value (image borders are always cuts).
fn e1_profiles(img: &[f32], w: usize, h: usize) -> (Vec<f64>, Vec<f64>) {
    let mut e1x = vec![0f64; w + 1];
    let mut e1y = vec![0f64; h + 1];
    // dx: sqrt(sum_c (img[y][x+1]-img[y][x])^2), summed over rows (f32 sums)
    let mut accx = vec![0f64; w.saturating_sub(1)];
    for y in 0..h {
        let r = y * w * 4;
        for x in 0..w - 1 {
            let mut s = 0f32;
            for c in 0..4 {
                let d = img[r + (x + 1) * 4 + c] - img[r + x * 4 + c];
                s += d * d;
            }
            accx[x] += (s.sqrt()) as f64;
        }
    }
    for x in 0..w - 1 {
        e1x[x + 1] = accx[x] as f32 as f64;
    }
    for y in 0..h - 1 {
        let r0 = y * w * 4;
        let r1 = (y + 1) * w * 4;
        let mut acc = 0f64;
        for x in 0..w {
            let mut s = 0f32;
            for c in 0..4 {
                let d = img[r1 + x * 4 + c] - img[r0 + x * 4 + c];
                s += d * d;
            }
            acc += (s.sqrt()) as f64;
        }
        e1y[y + 1] = acc as f32 as f64;
    }
    let mx = e1x[1..w].iter().cloned().fold(0.0, f64::max);
    let my = e1y[1..h].iter().cloned().fold(0.0, f64::max);
    let top = mx.max(my).max(1.0);
    e1x[0] = top;
    e1x[w] = top;
    e1y[0] = top;
    e1y[h] = top;
    (e1x, e1y)
}

/// Gradient/curvature magnitude maps: dqx (H, W-1), dqy (H-1, W) from the
/// quantized image; cox (H, W-2), coy (H-2, W) from the original.
pub struct GradMaps {
    pub dqx: Vec<f32>,
    pub dqy: Vec<f32>,
    pub cox: Vec<f32>,
    pub coy: Vec<f32>,
    pub w: usize,
    pub h: usize,
}

fn grad_maps(orig: &[f32], quant: &[f32], w: usize, h: usize) -> GradMaps {
    let mut dqx = vec![0f32; h * (w - 1)];
    let mut dqy = vec![0f32; (h - 1) * w];
    let mut cox = vec![0f32; h * (w - 2)];
    let mut coy = vec![0f32; (h - 2) * w];
    for y in 0..h {
        let r = y * w * 4;
        for x in 0..w - 1 {
            let mut s = 0f32;
            for c in 0..4 {
                let d = quant[r + (x + 1) * 4 + c] - quant[r + x * 4 + c];
                s += d * d;
            }
            dqx[y * (w - 1) + x] = s.sqrt();
        }
        for x in 0..w - 2 {
            let mut s = 0f32;
            for c in 0..4 {
                let d = orig[r + (x + 2) * 4 + c] - 2.0 * orig[r + (x + 1) * 4 + c]
                    + orig[r + x * 4 + c];
                s += d * d;
            }
            cox[y * (w - 2) + x] = s.sqrt();
        }
    }
    for y in 0..h - 1 {
        let r0 = y * w * 4;
        let r1 = (y + 1) * w * 4;
        for x in 0..w {
            let mut s = 0f32;
            for c in 0..4 {
                let d = quant[r1 + x * 4 + c] - quant[r0 + x * 4 + c];
                s += d * d;
            }
            dqy[y * w + x] = s.sqrt();
        }
    }
    for y in 0..h - 2 {
        let r0 = y * w * 4;
        let r1 = (y + 1) * w * 4;
        let r2 = (y + 2) * w * 4;
        for x in 0..w {
            let mut s = 0f32;
            for c in 0..4 {
                let d = orig[r2 + x * 4 + c] - 2.0 * orig[r1 + x * 4 + c] + orig[r0 + x * 4 + c];
                s += d * d;
            }
            coy[y * w + x] = s.sqrt();
        }
    }
    GradMaps {
        dqx,
        dqy,
        cox,
        coy,
        w,
        h,
    }
}

// ---------------------------------------------------------------- jpeg

pub fn jpeg_lattice_strength(profile: &[f64]) -> f64 {
    let norm = gaussian_filter1d(&normalise(profile), 0.6);
    let n = norm.len() - 1;
    let on: Vec<usize> = (8..n.saturating_sub(7)).step_by(8).collect();
    let off: Vec<usize> = (4..n.saturating_sub(3)).step_by(8).collect();
    if on.len() < 4 || off.len() < 4 {
        return 0.0;
    }
    let interior = &norm[1..n];
    let bm = interior.iter().sum::<f64>() / interior.len() as f64;
    let var = interior.iter().map(|&v| (v - bm) * (v - bm)).sum::<f64>() / interior.len() as f64;
    let bs = var.sqrt() + 1e-9;
    let z_on = (on.iter().map(|&i| norm[i]).sum::<f64>() / on.len() as f64 - bm)
        / (bs / (on.len() as f64).sqrt());
    let z_off = (off.iter().map(|&i| norm[i]).sum::<f64>() / off.len() as f64 - bm)
        / (bs / (off.len() as f64).sqrt());
    z_on - z_off.max(0.0)
}

pub fn notch_jpeg(profile: &[f64]) -> Vec<f64> {
    let n = profile.len() - 1;
    let width = 1.0;
    let mask: Vec<bool> = (0..=n)
        .map(|p| {
            if p == 0 || p == n {
                return false;
            }
            let m = (p % 8).min(8 - (p % 8)) as f64;
            m <= width
        })
        .collect();
    if !mask.iter().any(|&m| m) || mask.iter().all(|&m| m) {
        return profile.to_vec();
    }
    let good_x: Vec<f64> = (0..=n).filter(|&i| !mask[i]).map(|i| i as f64).collect();
    let good_y: Vec<f64> = (0..=n).filter(|&i| !mask[i]).map(|i| profile[i]).collect();
    (0..=n)
        .map(|i| {
            if mask[i] {
                interp(i as f64, &good_x, &good_y)
            } else {
                profile[i]
            }
        })
        .collect()
}

// ---------------------------------------------------------------- channels

/// Rayleigh phase-coherence score of profile peaks against a lattice
/// (channels._rayleigh_score on a pre-normalised profile).
pub fn rayleigh_score(profile_norm: &[f64], step: f64) -> f64 {
    let n = profile_norm.len() - 1;
    if step < 2.0 || step > n as f64 / 4.0 {
        return 0.0;
    }
    let dist = ((step * 0.4) as i64).max(1) as f64;
    let (pk, hts) = find_peaks(&profile_norm[1..n], 0.15, dist);
    if pk.len() < 5 {
        return 0.0;
    }
    let p: Vec<f64> = pk.iter().map(|&v| (v + 1) as f64).collect();
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut re = 0f64;
    let mut im = 0f64;
    let mut hsum = 0f64;
    let mut h2sum = 0f64;
    for (pi, &pos) in p.iter().enumerate() {
        let ang = two_pi * pos / step;
        re += hts[pi] * ang.cos();
        im += hts[pi] * ang.sin();
        hsum += hts[pi];
        h2sum += hts[pi] * hts[pi];
    }
    let rmag = (re * re + im * im).sqrt();
    let big_r = rmag / hsum;
    let n_eff = hsum * hsum / h2sum;
    let phase = ((im.atan2(re)) / two_pi * step).rem_euclid(step);
    let mut slots: Vec<i64> = Vec::new();
    for &pos in &p {
        let slot = ((pos - phase) / step).round_ties_even() as i64;
        if (pos - (phase + slot as f64 * step)).abs() < 0.35 * step {
            slots.push(slot);
        }
    }
    slots.sort_unstable();
    slots.dedup();
    let n_slots = std::cmp::max(1, (n as f64 / step) as i64 - 1);
    let occupancy = (slots.len() as f64 / n_slots as f64).min(1.0);
    (2.0 * n_eff).sqrt() * big_r * occupancy
}

/// Tile peak lists for a gradient map (pre-oriented: peaks along x).
/// offset mirrors the +1/+... position shift of the diff maps.
pub struct Tile {
    pub pos: Vec<f64>,
    pub heights: Vec<f64>,
    pub extent: usize,
}

pub fn tile_peaks(dmap: &[f32], rows: usize, cols: usize, offset: usize) -> Vec<Tile> {
    let tw = (cols / 6).clamp(48, 192);
    let th = (rows / 6).clamp(48, 192);
    let mut tiles = Vec::new();
    let mut y0 = 0usize;
    let ymax = std::cmp::max(rows as i64 - th as i64 / 2, 1) as usize;
    while y0 < ymax {
        let mut x0 = 0usize;
        let xmax = std::cmp::max(cols as i64 - tw as i64 / 2, 1) as usize;
        while x0 < xmax {
            let y1 = (y0 + th).min(rows);
            let x1 = (x0 + tw).min(cols);
            if y1 > y0 && x1 > x0 {
                let mut prof = vec![0f64; x1 - x0];
                for y in y0..y1 {
                    for x in x0..x1 {
                        prof[x - x0] += dmap[y * cols + x] as f64;
                    }
                }
                // f32-round the summed profile like numpy's f32 sum
                for v in prof.iter_mut() {
                    *v = *v as f32 as f64;
                }
                let scale = percentile(&prof, 95.0);
                if scale > 0.0 {
                    let norm: Vec<f64> =
                        prof.iter().map(|&v| (v / scale).clamp(0.0, 1.5)).collect();
                    let (pk, hts) = find_peaks(&norm, 0.2, 2.0);
                    if pk.len() >= 4 {
                        tiles.push(Tile {
                            pos: pk.iter().map(|&p| (p + x0 + offset) as f64).collect(),
                            heights: hts,
                            extent: prof.len(),
                        });
                    }
                }
            }
            x0 += tw;
        }
        y0 += th;
    }
    if tiles.len() > 360 {
        let idx: Vec<usize> = (0..360)
            .map(|i| (i as f64 * (tiles.len() as f64 - 1.0) / 359.0) as usize)
            .collect();
        tiles = idx
            .into_iter()
            .map(|i| Tile {
                pos: tiles[i].pos.clone(),
                heights: tiles[i].heights.clone(),
                extent: tiles[i].extent,
            })
            .collect();
    }
    tiles
}

/// Stouffer-combined per-tile Rayleigh coherence at `step`.
pub fn tiles_ray_z(tiles: &[Tile], step: f64) -> f64 {
    if tiles.is_empty() || step < 2.0 {
        return 0.0;
    }
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut zs: Vec<f64> = Vec::new();
    for t in tiles {
        if step > t.extent as f64 / 3.0 || t.pos.len() < 4 {
            continue;
        }
        let mut re = 0f64;
        let mut im = 0f64;
        let mut hsum = 0f64;
        let mut h2sum = 0f64;
        for (i, &pos) in t.pos.iter().enumerate() {
            let ang = two_pi * pos / step;
            re += t.heights[i] * ang.cos();
            im += t.heights[i] * ang.sin();
            hsum += t.heights[i];
            h2sum += t.heights[i] * t.heights[i];
        }
        let big_r = (re * re + im * im).sqrt() / (hsum + 1e-9);
        let n_eff = hsum * hsum / (h2sum + 1e-9);
        let phase = (im.atan2(re) / two_pi * step).rem_euclid(step);
        let mut slots: Vec<i64> = Vec::new();
        for &pos in &t.pos {
            let slot = ((pos - phase) / step).round_ties_even() as i64;
            if (pos - (phase + slot as f64 * step)).abs() < 0.35 * step {
                slots.push(slot);
            }
        }
        slots.sort_unstable();
        slots.dedup();
        let n_slots = std::cmp::max(1, (t.extent as f64 / step) as i64);
        let occ = (slots.len() as f64 / n_slots as f64).min(1.0);
        zs.push((2.0 * n_eff).sqrt() * big_r * occ - 1.0);
    }
    if zs.len() < 2 {
        return 0.0;
    }
    zs.iter().sum::<f64>() / (zs.len() as f64).sqrt()
}

// ---------------------------------------------------------------- spectrum

pub struct Spectrum {
    pub freqs: Vec<f64>,
    pub power: Vec<f64>,
    pub bg: Vec<f64>,
}

/// Welch-averaged power spectrum of gradient scanline groups (row_group=4,
/// Hann window, hop win/2), pre-oriented map (rows x cols).
pub fn axis_spectrum(
    dmap: &[f32],
    rows: usize,
    cols: usize,
    planner: &mut FftPlanner<f64>,
) -> Spectrum {
    let win = cols.min(1024);
    if win < 32 {
        return Spectrum {
            freqs: vec![0.0],
            power: vec![0.0],
            bg: vec![0.0],
        };
    }
    // np.hanning(win): 0.5 - 0.5*cos(2*pi*i/(win-1))
    let window: Vec<f64> = (0..win)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (win as f64 - 1.0)).cos())
        .collect();
    let hop = std::cmp::max(win / 2, 1);
    let fft = planner.plan_fft_forward(win);
    let nbins = win / 2 + 1;
    let mut specs = vec![0f64; nbins];
    let mut count = 0usize;
    let row_group = 4;
    let mut buf = vec![Complex::new(0f64, 0f64); win];
    let mut y0 = 0usize;
    while y0 + row_group <= rows {
        // group profile: f32 sums per numpy
        let mut prof = vec![0f64; cols];
        for y in y0..y0 + row_group {
            for x in 0..cols {
                prof[x] += dmap[y * cols + x] as f64;
            }
        }
        for v in prof.iter_mut() {
            *v = *v as f32 as f64;
        }
        let mean = prof.iter().sum::<f64>() / cols as f64;
        for v in prof.iter_mut() {
            *v -= mean;
        }
        let mut x0 = 0usize;
        while x0 + win <= cols {
            for i in 0..win {
                buf[i] = Complex::new(prof[x0 + i] * window[i], 0.0);
            }
            fft.process(&mut buf);
            for k in 0..nbins {
                specs[k] += buf[k].norm_sqr();
            }
            count += 1;
            x0 += hop;
        }
        y0 += row_group;
    }
    if count == 0 {
        return Spectrum {
            freqs: vec![0.0],
            power: vec![0.0],
            bg: vec![0.0],
        };
    }
    let power: Vec<f64> = specs.iter().map(|&p| p / count as f64).collect();
    let freqs: Vec<f64> = (0..nbins).map(|k| k as f64 / win as f64).collect();
    // background: median filter, size k odd >= 5
    let mut ksz = std::cmp::max(5, power.len() / 24);
    if ksz % 2 == 0 {
        ksz += 1;
    }
    let bg = median_filter1d(&power, ksz);
    Spectrum { freqs, power, bg }
}

pub fn spectral_z(sp: &Spectrum, step: f64) -> f64 {
    if step < 2.0 || sp.freqs.len() < 8 {
        return 0.0;
    }
    let f = 1.0 / step;
    if f <= sp.freqs[1] || f >= *sp.freqs.last().unwrap() {
        return 0.0;
    }
    // np.searchsorted(freqs, f) side=left
    let mut idx = sp.freqs.partition_point(|&v| v < f);
    if idx >= sp.power.len() {
        idx = sp.power.len() - 1;
    }
    let lo = std::cmp::max(1, idx as i64 - 1) as usize;
    let hi = std::cmp::min(sp.power.len() - 1, idx + 1);
    let p = sp.power[lo..=hi].iter().cloned().fold(f64::MIN, f64::max);
    let b = sp.bg[idx] + 1e-12;
    let ratio = (p / b).max(1e-6);
    6.0 * ratio.log10()
}

// ---------------------------------------------------------------- evidence

pub struct AxisEvidence {
    pub e1: Vec<f64>,
    pub tiles1: Vec<Tile>,
    pub tiles2: Vec<Tile>,
    pub spec1: Spectrum,
    pub extent: usize,
}

pub struct Evidence {
    pub x: AxisEvidence,
    pub y: AxisEvidence,
    pub jpeg_z: f64,
}

fn transpose_map(m: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * cols];
    for y in 0..rows {
        for x in 0..cols {
            out[x * rows + y] = m[y * cols + x];
        }
    }
    out
}

/// build_evidence(rgba, lean=True): median blur -> kmeans16 -> profiles,
/// jpeg notch, grad maps, tile peaks, E1 spectra.
pub fn build_evidence(rgba: &[u8], w: usize, h: usize) -> Evidence {
    // cv2.medianBlur(rgba, 3) on all 4 channels
    let base = crate::runlengths::prep_u8(rgba, w, h);
    let quantized = kmeans_quantize_rgba(&base, w, h, 16);

    let img_o = flatten_channels(rgba, w, h);
    let img_q = flatten_channels(&quantized, w, h);

    let (mut e1x, mut e1y) = e1_profiles(&img_q, w, h);
    let jpeg_z = jpeg_lattice_strength(&e1x).max(jpeg_lattice_strength(&e1y));
    if jpeg_z > 5.0 {
        e1x = notch_jpeg(&e1x);
        e1y = notch_jpeg(&e1y);
    }

    let gm = grad_maps(&img_o, &img_q, w, h);
    drop(img_o);
    drop(img_q);

    // x axis: dqx (h, w-1) offset 1; cox (h, w-2) offset 1
    // y axis: transpose dqy/coy so peaks run along the y extent
    let dqy_t = transpose_map(&gm.dqy, h - 1, w);
    let coy_t = transpose_map(&gm.coy, h - 2, w);

    let ((tiles1x, tiles2x, spec1x), (tiles1y, tiles2y, spec1y)) = rayon::join(
        || {
            let mut planner = FftPlanner::new();
            (
                tile_peaks(&gm.dqx, h, w - 1, 1),
                tile_peaks(&gm.cox, h, w - 2, 1),
                axis_spectrum(&gm.dqx, h, w - 1, &mut planner),
            )
        },
        || {
            let mut planner = FftPlanner::new();
            (
                tile_peaks(&dqy_t, w, h - 1, 1),
                tile_peaks(&coy_t, w, h - 2, 1),
                axis_spectrum(&dqy_t, w, h - 1, &mut planner),
            )
        },
    );

    Evidence {
        x: AxisEvidence {
            e1: e1x,
            tiles1: tiles1x,
            tiles2: tiles2x,
            spec1: spec1x,
            extent: w,
        },
        y: AxisEvidence {
            e1: e1y,
            tiles1: tiles1y,
            tiles2: tiles2y,
            spec1: spec1y,
            extent: h,
        },
        jpeg_z,
    }
}

/// Fused curve over the ladder: equal-weight sum of per-scan max-normalised
/// {ray_e1, tile_e1, tile_e2, spec_e1} (fusion.py WEIGHTS).
pub fn fused_curve(ev_axis: &AxisEvidence, steps: &[f64]) -> Vec<f64> {
    let e1_norm = normalise(&ev_axis.e1);
    let cols: Vec<Vec<f64>> = vec![
        steps
            .par_iter()
            .map(|&s| rayleigh_score(&e1_norm, s))
            .collect(),
        steps
            .par_iter()
            .map(|&s| tiles_ray_z(&ev_axis.tiles1, s))
            .collect(),
        steps
            .par_iter()
            .map(|&s| tiles_ray_z(&ev_axis.tiles2, s))
            .collect(),
        steps
            .par_iter()
            .map(|&s| spectral_z(&ev_axis.spec1, s))
            .collect(),
    ];
    let mut fused = vec![0f64; steps.len()];
    for col in &cols {
        let m = col.iter().cloned().map(|v| v.max(0.0)).fold(0.0, f64::max);
        if m <= 1e-9 {
            continue;
        }
        for i in 0..steps.len() {
            fused[i] += col[i].max(0.0) / m;
        }
    }
    fused
}
