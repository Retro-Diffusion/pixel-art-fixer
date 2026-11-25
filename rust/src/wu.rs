//! Wu (1992) color quantizer + elbow color-count detection.
//! Mirrors detector/wu.py. Deterministic (no RNG) so it matches the Python
//! reference exactly. Used to snap the per-cell mode reconstruction onto a
//! small optimal pixel-art palette.

const SIDE: usize = 33; // 32 bins + 1 SAT pad
const SHIFT: u32 = 3; // 8 -> 5 bits

#[inline]
fn at(i: usize, j: usize, k: usize) -> usize {
    (i * SIDE + j) * SIDE + k
}

struct Moments {
    wt: Vec<f64>,
    mr: Vec<f64>,
    mg: Vec<f64>,
    mb: Vec<f64>,
    m2: Vec<f64>,
}

type Box6 = (usize, usize, usize, usize, usize, usize);

fn build_moments(rgb: &[[u8; 3]]) -> Moments {
    let n = SIDE * SIDE * SIDE;
    let mut wt = vec![0f64; n];
    let mut mr = vec![0f64; n];
    let mut mg = vec![0f64; n];
    let mut mb = vec![0f64; n];
    let mut m2 = vec![0f64; n];
    for px in rgb {
        let r = (px[0] >> SHIFT) as usize + 1;
        let g = (px[1] >> SHIFT) as usize + 1;
        let b = (px[2] >> SHIFT) as usize + 1;
        let idx = at(r, g, b);
        let (rf, gf, bf) = (px[0] as f64, px[1] as f64, px[2] as f64);
        wt[idx] += 1.0;
        mr[idx] += rf;
        mg[idx] += gf;
        mb[idx] += bf;
        m2[idx] += rf * rf + gf * gf + bf * bf;
    }
    // 3-D cumulative (summed-area) along all three axes, matching
    // numpy's cumsum(0).cumsum(1).cumsum(2) order.
    for arr in [&mut wt, &mut mr, &mut mg, &mut mb, &mut m2] {
        // axis 0
        for i in 1..SIDE {
            for j in 0..SIDE {
                for k in 0..SIDE {
                    arr[at(i, j, k)] += arr[at(i - 1, j, k)];
                }
            }
        }
        // axis 1
        for i in 0..SIDE {
            for j in 1..SIDE {
                for k in 0..SIDE {
                    arr[at(i, j, k)] += arr[at(i, j - 1, k)];
                }
            }
        }
        // axis 2
        for i in 0..SIDE {
            for j in 0..SIDE {
                for k in 1..SIDE {
                    arr[at(i, j, k)] += arr[at(i, j, k - 1)];
                }
            }
        }
    }
    Moments { wt, mr, mg, mb, m2 }
}

fn box_sum(m: &[f64], bx: Box6) -> f64 {
    let (r0, r1, g0, g1, b0, b1) = bx;
    m[at(r1, g1, b1)] - m[at(r1, g1, b0)] - m[at(r1, g0, b1)] + m[at(r1, g0, b0)]
        - m[at(r0, g1, b1)]
        + m[at(r0, g1, b0)]
        + m[at(r0, g0, b1)]
        - m[at(r0, g0, b0)]
}

fn variance(bx: Box6, m: &Moments) -> f64 {
    let dw = box_sum(&m.wt, bx);
    if dw <= 0.0 {
        return 0.0;
    }
    let dr = box_sum(&m.mr, bx);
    let dg = box_sum(&m.mg, bx);
    let db = box_sum(&m.mb, bx);
    box_sum(&m.m2, bx) - (dr * dr + dg * dg + db * db) / dw
}

/// best split along axis (0=r,1=g,2=b): (max between-class value, cut) or -1.
fn maximize(bx: Box6, m: &Moments, axis: usize) -> (f64, i64) {
    let (r0, r1, g0, g1, b0, b1) = bx;
    let whole_r = box_sum(&m.mr, bx);
    let whole_g = box_sum(&m.mg, bx);
    let whole_b = box_sum(&m.mb, bx);
    let whole_w = box_sum(&m.wt, bx);
    let (lo, hi) = match axis {
        0 => (r0 + 1, r1),
        1 => (g0 + 1, g1),
        _ => (b0 + 1, b1),
    };
    let mut best = 0.0f64;
    let mut cut: i64 = -1;
    for i in lo..hi {
        let half = match axis {
            0 => (r0, i, g0, g1, b0, b1),
            1 => (r0, r1, g0, i, b0, b1),
            _ => (r0, r1, g0, g1, b0, i),
        };
        let hw = box_sum(&m.wt, half);
        if hw == 0.0 {
            continue;
        }
        let hr = box_sum(&m.mr, half);
        let hg = box_sum(&m.mg, half);
        let hb = box_sum(&m.mb, half);
        let mut temp = (hr * hr + hg * hg + hb * hb) / hw;
        let (rr, rg, rb, rw) = (whole_r - hr, whole_g - hg, whole_b - hb, whole_w - hw);
        if rw == 0.0 {
            continue;
        }
        temp += (rr * rr + rg * rg + rb * rb) / rw;
        if temp > best {
            best = temp;
            cut = i as i64;
        }
    }
    (best, cut)
}

fn cut_box(bx: Box6, m: &Moments) -> Option<(Box6, Box6)> {
    // choose the axis with the largest maximize() value (ties -> lower axis,
    // matching Python's sort(reverse=True) on (value, axis, cut))
    let mut best_ax = 0usize;
    let mut best_val = f64::NEG_INFINITY;
    let mut best_cut = -1i64;
    for ax in 0..3 {
        let (v, c) = maximize(bx, m, ax);
        if v > best_val {
            best_val = v;
            best_ax = ax;
            best_cut = c;
        }
    }
    if best_cut < 0 {
        return None;
    }
    let (r0, r1, g0, g1, b0, b1) = bx;
    let c = best_cut as usize;
    Some(match best_ax {
        0 => ((r0, c, g0, g1, b0, b1), (c, r1, g0, g1, b0, b1)),
        1 => ((r0, r1, g0, c, b0, b1), (r0, r1, c, g1, b0, b1)),
        _ => ((r0, r1, g0, g1, b0, c), (r0, r1, g0, g1, c, b1)),
    })
}

/// Wu-quantize to <=k colors. Returns (palette [0,1], labels into palette).
pub fn quantize(rgb: &[[u8; 3]], k: usize) -> (Vec<[f64; 3]>, Vec<usize>) {
    if rgb.is_empty() {
        return (vec![[0.0; 3]], vec![]);
    }
    let m = build_moments(rgb);
    let whole: Box6 = (0, SIDE - 1, 0, SIDE - 1, 0, SIDE - 1);
    let mut boxes = vec![whole];
    let mut vars = vec![variance(whole, &m)];
    while boxes.len() < k {
        // highest-variance box (first max, matching np.argmax)
        let mut i = 0usize;
        for j in 1..vars.len() {
            if vars[j] > vars[i] {
                i = j;
            }
        }
        if vars[i] <= 0.0 {
            break;
        }
        match cut_box(boxes[i], &m) {
            None => vars[i] = 0.0,
            Some((a, b)) => {
                boxes[i] = a;
                boxes.push(b);
                vars[i] = variance(a, &m);
                vars.push(variance(b, &m));
            }
        }
    }
    let mut pal: Vec<[f64; 3]> = Vec::new();
    for &bx in &boxes {
        let dw = box_sum(&m.wt, bx);
        if dw <= 0.0 {
            continue;
        }
        pal.push([
            box_sum(&m.mr, bx) / dw / 255.0,
            box_sum(&m.mg, bx) / dw / 255.0,
            box_sum(&m.mb, bx) / dw / 255.0,
        ]);
    }
    if pal.is_empty() {
        let mut s = [0f64; 3];
        for p in rgb {
            for c in 0..3 {
                s[c] += p[c] as f64;
            }
        }
        pal.push([
            s[0] / rgb.len() as f64 / 255.0,
            s[1] / rgb.len() as f64 / 255.0,
            s[2] / rgb.len() as f64 / 255.0,
        ]);
    }
    // assign each pixel to nearest palette color
    let labels: Vec<usize> = rgb
        .iter()
        .map(|px| {
            let p = [
                px[0] as f64 / 255.0,
                px[1] as f64 / 255.0,
                px[2] as f64 / 255.0,
            ];
            let mut best = 0usize;
            let mut bd = f64::INFINITY;
            for (ci, c) in pal.iter().enumerate() {
                let d = (p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2) + (p[2] - c[2]).powi(2);
                if d < bd {
                    bd = d;
                    best = ci;
                }
            }
            best
        })
        .collect();
    (pal, labels)
}

/// K by the knee of the Wu RMSE curve on a clean image (mirrors
/// detector.wu.elbow_color_count).
pub fn elbow_color_count(rgb: &[[u8; 3]], max_colors: usize) -> usize {
    if rgb.is_empty() {
        return 2;
    }
    let ks: Vec<usize> = [8, 12, 16, 20, 24, 32, 40, 48, 64]
        .into_iter()
        .filter(|&k| k <= max_colors)
        .collect();
    let ks = if ks.is_empty() { vec![max_colors] } else { ks };
    let mut errs = Vec::with_capacity(ks.len());
    for &k in &ks {
        let (pal, lab) = quantize(rgb, k);
        let mut se = 0f64;
        for (i, px) in rgb.iter().enumerate() {
            let c = pal[lab[i]];
            for ch in 0..3 {
                let d = c[ch] * 255.0 - px[ch] as f64;
                se += d * d;
            }
        }
        errs.push((se / rgb.len() as f64).sqrt());
    }
    let thr = 0.055 * errs[0].max(1e-9);
    for i in 1..ks.len() {
        if errs[i - 1] - errs[i] < thr {
            return ks[i - 1];
        }
    }
    *ks.last().unwrap()
}
