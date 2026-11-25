//! Deterministic k-means++ used by the fusion evidence (quantize.py) and
//! reconsearch (_quantize). cv2.kmeans depends on OpenCV's global RNG, so
//! bit-parity is impossible; this is a faithful-criteria replacement with
//! a fixed-seed RNG. Downstream scores are aggregates over many steps and
//! are robust to the small clustering differences.

/// xorshift64* — deterministic, decent quality, no deps.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.max(1))
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_f64() * n as f64) as usize % n.max(1)
    }
}

/// Evenly-spaced deterministic sample of up to `max_n` row indices.
pub fn even_sample(n: usize, max_n: usize) -> Vec<usize> {
    if n <= max_n {
        (0..n).collect()
    } else {
        (0..max_n)
            .map(|i| ((i as f64) * (n as f64 - 1.0) / (max_n as f64 - 1.0)) as usize)
            .collect()
    }
}

fn dist2(a: &[f32; 3], b: &[f32; 3]) -> f64 {
    let mut s = 0f64;
    for c in 0..3 {
        let d = (a[c] - b[c]) as f64;
        s += d * d;
    }
    s
}

/// k-means++ init + Lloyd iterations; returns (centers, inertia).
fn kmeans_once(
    points: &[[f32; 3]],
    k: usize,
    max_iter: usize,
    eps: f64,
    rng: &mut Rng,
) -> (Vec<[f32; 3]>, f64) {
    let n = points.len();
    let mut centers: Vec<[f32; 3]> = Vec::with_capacity(k);
    centers.push(points[rng.below(n)]);
    let mut d2: Vec<f64> = points.iter().map(|p| dist2(p, &centers[0])).collect();
    while centers.len() < k {
        let total: f64 = d2.iter().sum();
        let mut pick = 0usize;
        if total > 0.0 {
            let target = rng.next_f64() * total;
            let mut acc = 0f64;
            for i in 0..n {
                acc += d2[i];
                if acc >= target {
                    pick = i;
                    break;
                }
            }
        } else {
            pick = rng.below(n);
        }
        let c = points[pick];
        centers.push(c);
        for i in 0..n {
            let d = dist2(&points[i], &c);
            if d < d2[i] {
                d2[i] = d;
            }
        }
    }

    let mut labels = vec![0u32; n];
    for _ in 0..max_iter {
        // assign
        for i in 0..n {
            let mut best = 0usize;
            let mut bd = f64::INFINITY;
            for (ci, c) in centers.iter().enumerate() {
                let d = dist2(&points[i], c);
                if d < bd {
                    bd = d;
                    best = ci;
                }
            }
            labels[i] = best as u32;
        }
        // update
        let mut sums = vec![[0f64; 3]; k];
        let mut cnts = vec![0usize; k];
        for i in 0..n {
            let l = labels[i] as usize;
            cnts[l] += 1;
            for c in 0..3 {
                sums[l][c] += points[i][c] as f64;
            }
        }
        let mut max_shift = 0f64;
        for ci in 0..k {
            if cnts[ci] == 0 {
                // OpenCV-style: reseed empty cluster at the farthest point
                let mut far = 0usize;
                let mut fd = -1f64;
                for i in 0..n {
                    let d = dist2(&points[i], &centers[labels[i] as usize]);
                    if d > fd {
                        fd = d;
                        far = i;
                    }
                }
                centers[ci] = points[far];
                max_shift = f64::INFINITY;
                continue;
            }
            let mut nc = [0f32; 3];
            for c in 0..3 {
                nc[c] = (sums[ci][c] / cnts[ci] as f64) as f32;
            }
            let shift = dist2(&nc, &centers[ci]);
            if shift > max_shift {
                max_shift = shift;
            }
            centers[ci] = nc;
        }
        if max_shift <= eps * eps {
            break;
        }
    }
    let mut inertia = 0f64;
    for i in 0..n {
        inertia += dist2(&points[i], &centers[labels[i] as usize]);
    }
    (centers, inertia)
}

/// Multi-attempt k-means (best inertia wins), fixed seed.
pub fn kmeans(
    points: &[[f32; 3]],
    k: usize,
    max_iter: usize,
    eps: f64,
    attempts: usize,
    seed: u64,
) -> Vec<[f32; 3]> {
    let mut rng = Rng::new(seed);
    let mut best: Option<(Vec<[f32; 3]>, f64)> = None;
    for _ in 0..attempts {
        let (c, inertia) = kmeans_once(points, k, max_iter, eps, &mut rng);
        if best.as_ref().map_or(true, |b| inertia < b.1) {
            best = Some((c, inertia));
        }
    }
    best.unwrap().0
}

/// quantize.py kmeans_quantize: k=16 on RGB u8, returns quantized RGBA.
pub fn kmeans_quantize_rgba(rgba: &[u8], w: usize, h: usize, k: usize) -> Vec<u8> {
    let n = w * h;
    let sample_idx = even_sample(n, 60_000);
    let sample: Vec<[f32; 3]> = sample_idx
        .iter()
        .map(|&i| {
            let p = i * 4;
            [rgba[p] as f32, rgba[p + 1] as f32, rgba[p + 2] as f32]
        })
        .collect();
    // k_eff = min(k, unique sample colors)
    let mut uniq: Vec<[u8; 3]> = sample_idx
        .iter()
        .map(|&i| {
            let p = i * 4;
            [rgba[p], rgba[p + 1], rgba[p + 2]]
        })
        .collect();
    uniq.sort_unstable();
    uniq.dedup();
    let k_eff = k.min(uniq.len());
    if k_eff <= 1 {
        return rgba.to_vec();
    }
    let centers = kmeans(&sample, k_eff, 12, 0.5, 2, 42);

    let mut out = rgba.to_vec();
    for i in 0..n {
        let p = i * 4;
        let px = [rgba[p] as f32, rgba[p + 1] as f32, rgba[p + 2] as f32];
        let mut best = 0usize;
        let mut bd = f64::INFINITY;
        for (ci, c) in centers.iter().enumerate() {
            let d = dist2(&px, c);
            if d < bd {
                bd = d;
                best = ci;
            }
        }
        for c in 0..3 {
            // np.rint = banker's
            out[p + c] = (centers[best][c] as f64)
                .round_ties_even()
                .clamp(0.0, 255.0) as u8;
        }
    }
    out
}

/// Adaptive structure-only K from coarse (4-bit) colour complexity of the
/// opaque pixels; mirrors detector.reconstruct.adaptive_k.
pub fn adaptive_k(rgba: &[u8], w: usize, h: usize, lo: usize, hi: usize, share: f64) -> usize {
    let mut cnt = vec![0u32; 4096];
    let mut total = 0u64;
    for i in 0..w * h {
        if rgba[i * 4 + 3] > 0 {
            let key = (((rgba[i * 4] >> 4) as usize) << 8)
                | (((rgba[i * 4 + 1] >> 4) as usize) << 4)
                | ((rgba[i * 4 + 2] >> 4) as usize);
            cnt[key] += 1;
            total += 1;
        }
    }
    if total == 0 {
        return lo;
    }
    let k = cnt
        .iter()
        .filter(|&&c| c as f64 / total as f64 >= share)
        .count();
    k.clamp(lo, hi)
}

/// k-means (sample for centroids, then assign every pixel) -> (labels, K).
/// Used by two-stage packing for the STRUCTURE quantisation.
pub fn kmeans_labels(rgba: &[u8], w: usize, h: usize, k: usize) -> (Vec<u32>, usize) {
    use rayon::prelude::*;
    let n = w * h;
    let opaque: Vec<usize> = (0..n).filter(|&i| rgba[i * 4 + 3] > 0).collect();
    let src: Vec<usize> = if opaque.is_empty() {
        (0..n).collect()
    } else {
        opaque
    };
    let sample_idx = even_sample(src.len(), 60_000);
    let sample: Vec<[f32; 3]> = sample_idx
        .iter()
        .map(|&si| {
            let i = src[si];
            [
                rgba[i * 4] as f32,
                rgba[i * 4 + 1] as f32,
                rgba[i * 4 + 2] as f32,
            ]
        })
        .collect();
    let k_eff = k.min(sample.len()).max(1);
    let centers = kmeans(&sample, k_eff, 15, 0.5, 1, 42);
    let kc = centers.len();
    let labels: Vec<u32> = (0..n)
        .into_par_iter()
        .map(|i| {
            let p = [
                rgba[i * 4] as f32,
                rgba[i * 4 + 1] as f32,
                rgba[i * 4 + 2] as f32,
            ];
            let mut best = 0u32;
            let mut bd = f32::INFINITY;
            for (ci, c) in centers.iter().enumerate() {
                let d = (p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2) + (p[2] - c[2]).powi(2);
                if d < bd {
                    bd = d;
                    best = ci as u32;
                }
            }
            best
        })
        .collect();
    (labels, kc)
}
