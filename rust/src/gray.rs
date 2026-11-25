//! Feature maps: gray conversion, median quantization, derivative maps.
//! Mirrors detector/autocorr.py (to_gray, median_quant, d1_along, d2_along).

/// Gray over white per alpha. rgba: interleaved u8, len = w*h*4.
pub fn to_gray(rgba: &[u8], w: usize, h: usize) -> Vec<f32> {
    let mut g = vec![0f32; w * h];
    for i in 0..w * h {
        let p = i * 4;
        let a = rgba[p + 3] as f32 / 255.0;
        let lum = 0.299 * rgba[p] as f32 + 0.587 * rgba[p + 1] as f32 + 0.114 * rgba[p + 2] as f32;
        g[i] = lum * a + 127.5 * (1.0 - a);
    }
    g
}

/// 3x3 median (border replicate) then quantize to steps of 12.
pub fn median_quant(g: &[f32], w: usize, h: usize) -> Vec<f32> {
    // numpy astype(uint8): truncation toward zero (g is within [0, 255.001])
    let u8v: Vec<u8> = g.iter().map(|&v| v as u8).collect();
    let mut out = vec![0f32; w * h];
    let mut buf = [0u8; 9];
    for y in 0..h {
        for x in 0..w {
            let mut k = 0;
            for dy in -1i64..=1 {
                let yy = (y as i64 + dy).clamp(0, h as i64 - 1) as usize;
                for dx in -1i64..=1 {
                    let xx = (x as i64 + dx).clamp(0, w as i64 - 1) as usize;
                    buf[k] = u8v[yy * w + xx];
                    k += 1;
                }
            }
            buf.sort_unstable();
            // np.round = half-to-even, not half-away-from-zero
            out[y * w + x] = (buf[4] as f32 / 12.0).round_ties_even() * 12.0;
        }
    }
    out
}

/// |d1| along x, padded to width w (row-major lines x extent).
pub fn d1x(g: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0f32; w * h];
    for y in 0..h {
        let r = y * w;
        for x in 0..w - 1 {
            out[r + x] = (g[r + x + 1] - g[r + x]).abs();
        }
    }
    out
}

/// |d2| along x, padded (1, 1).
pub fn d2x(g: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0f32; w * h];
    for y in 0..h {
        let r = y * w;
        for x in 1..w - 1 {
            out[r + x] = (g[r + x + 1] - 2.0 * g[r + x] + g[r + x - 1]).abs();
        }
    }
    out
}

pub fn transpose(a: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            out[x * h + y] = a[y * w + x];
        }
    }
    out
}
