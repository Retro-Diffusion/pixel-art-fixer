//! Consensus core. Mirrors detector/core.py: the fast mode (ac + rl with
//! the calibrated early exit, selfsim on disagreement) and the full mode
//! (stage-1 supermajority consensus over four voters, stage-2 arbitration
//! over fused evidence + square-packer + distillability).

use crate::autocorr;
use crate::fusionchan;
use crate::reconsearch;
use crate::runlengths;
use crate::selfsim;
use crate::varcontrast::{vc_z_of, CellVarContrast};

pub struct CoreResult {
    pub step_x: f64,
    pub step_y: f64,
    pub cols: i64,
    pub rows: i64,
    pub consensus: String,
}

struct Prop {
    step_x: f64,
    step_y: f64,
    cols: i64,
    rows: i64,
}

fn size_close(a: &Prop, b: &Prop) -> bool {
    let tol_c = std::cmp::max(1, (0.01 * b.cols as f64).round_ties_even() as i64);
    let tol_r = std::cmp::max(1, (0.01 * b.rows as f64).round_ties_even() as i64);
    (a.cols - b.cols).abs() <= tol_c && (a.rows - b.rows).abs() <= tol_r
}

/// mode="fast": ac + rl with calibrated early exit; selfsim only on
/// disagreement; first agreeing pair wins, else autocorr flagged lowconf.
pub fn detect_fast(rgba: &[u8], w: usize, h: usize) -> CoreResult {
    let (ac, rl) = rayon::join(
        || autocorr::detect(rgba, w, h),
        || runlengths::detect(rgba, w, h),
    );
    let p_ac = Prop {
        step_x: ac.step_x,
        step_y: ac.step_y,
        cols: ac.cols as i64,
        rows: ac.rows as i64,
    };
    let p_rl = Prop {
        step_x: rl.step_x,
        step_y: rl.step_y,
        cols: rl.cols,
        rows: rl.rows,
    };

    // calibrated early exit: rl comb peak S >= 0.30 was always correct on
    // the benchmark; with ac agreeing on size, selfsim adds nothing
    if size_close(&p_ac, &p_rl) && rl.score_x.min(rl.score_y) >= 0.30 {
        let cols = ((p_ac.cols + p_rl.cols) as f64 / 2.0).round_ties_even() as i64;
        let rows = ((p_ac.rows + p_rl.rows) as f64 / 2.0).round_ties_even() as i64;
        return CoreResult {
            step_x: w as f64 / cols as f64,
            step_y: h as f64 / rows as f64,
            cols,
            rows,
            consensus: "fast:ac+rl(S)".into(),
        };
    }

    let ss = selfsim::detect(rgba, w, h);
    let p_ss = Prop {
        step_x: ss.step_x,
        step_y: ss.step_y,
        cols: ss.cols,
        rows: ss.rows,
    };

    // first size-close pair in reference order wins (adopting the first
    // member's steps), else autocorr flagged low-confidence
    let named: [(&str, &Prop); 3] = [("ac", &p_ac), ("rl", &p_rl), ("ss", &p_ss)];
    for i in 0..named.len() {
        for j in i + 1..named.len() {
            let (na, a) = named[i];
            let (nb, b) = named[j];
            if size_close(a, b) {
                return CoreResult {
                    step_x: a.step_x,
                    step_y: a.step_y,
                    cols: a.cols,
                    rows: a.rows,
                    consensus: format!("fastmode:{}+{}", na, nb),
                };
            }
        }
    }
    CoreResult {
        step_x: p_ac.step_x,
        step_y: p_ac.step_y,
        cols: p_ac.cols,
        rows: p_ac.rows,
        consensus: "fastmode:lowconf".into(),
    }
}

// ---------------------------------------------------------------- full mode

const AGREE_TOL: f64 = 0.02;
const AGREE_BONUS: f64 = 0.25;
const VC_W: f64 = 0.20;
const SMALLEST_QUALIFIED: f64 = 0.78;

/// detail-scale bound from the pooled ACF central peak half-width
fn acf_width(ac: &[f64]) -> f64 {
    let tail: &[f64] = if ac.len() > 40 {
        &ac[32..ac.len().min(96)]
    } else {
        &ac[ac.len() / 2..]
    };
    let base = if tail.is_empty() {
        0.0
    } else {
        crate::sigproc::median(tail)
    };
    let c0 = (ac[0] - base).max(1e-9);
    for lag in 1..ac.len().min(64) {
        if ac[lag] - base < 0.30 * c0 {
            return lag as f64;
        }
    }
    64.0
}

/// int(np.median(ints)): middle for odd, truncated mean of middles for even
fn int_median(vals: &[i64]) -> i64 {
    let mut v = vals.to_vec();
    v.sort_unstable();
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        ((v[n / 2 - 1] + v[n / 2]) as f64 / 2.0) as i64
    }
}

/// mode="full": consensus + arbitration (best accuracy).
pub fn detect_full(rgba: &[u8], w: usize, h: usize) -> CoreResult {
    // ---- cheap trio (keep the autocorr internals for arbitration)
    let (pre, rl) = rayon::join(
        || autocorr::prepare(rgba, w, h),
        || runlengths::detect(rgba, w, h),
    );
    let ac = autocorr::detect_pre(&pre, w, h);
    let p_ac = Prop {
        step_x: ac.step_x,
        step_y: ac.step_y,
        cols: ac.cols as i64,
        rows: ac.rows as i64,
    };
    let p_rl = Prop {
        step_x: rl.step_x,
        step_y: rl.step_y,
        cols: rl.cols,
        rows: rl.rows,
    };
    if size_close(&p_ac, &p_rl) && rl.score_x.min(rl.score_y) >= 0.30 {
        let cols = ((p_ac.cols + p_rl.cols) as f64 / 2.0).round_ties_even() as i64;
        let rows = ((p_ac.rows + p_rl.rows) as f64 / 2.0).round_ties_even() as i64;
        return CoreResult {
            step_x: w as f64 / cols as f64,
            step_y: h as f64 / rows as f64,
            cols,
            rows,
            consensus: "fast:ac+rl(S)".into(),
        };
    }
    let ss = selfsim::detect(rgba, w, h);
    let p_ss = Prop {
        step_x: ss.step_x,
        step_y: ss.step_y,
        cols: ss.cols,
        rows: ss.rows,
    };

    let aspect_ok = |p: &Prop| -> bool {
        let r = (p.cols as f64 / (p.rows.max(1)) as f64) / (w as f64 / h as f64);
        r.ln().abs() < 0.35
    };

    // ---- fast path: three cheap detectors already agree on the size
    {
        let trio = [("ac", &p_ac), ("rl", &p_rl), ("ss", &p_ss)];
        let agree: Vec<&str> = trio
            .iter()
            .filter(|(_, p)| size_close(p, &p_ac))
            .map(|(n, _)| *n)
            .collect();
        if agree.len() >= 3 && aspect_ok(&p_ac) {
            let cols = int_median(&[p_ac.cols, p_rl.cols, p_ss.cols]);
            let rows = int_median(&[p_ac.rows, p_rl.rows, p_ss.rows]);
            return CoreResult {
                step_x: w as f64 / cols as f64,
                step_y: h as f64 / rows as f64,
                cols,
                rows,
                consensus: format!("fast:{}", agree.join("+")),
            };
        }
    }

    // ---- heavy evidence: fused channels, square packer, distillability
    let ((ev, vc_data), recon) = rayon::join(
        || {
            rayon::join(
                || fusionchan::build_evidence(rgba, w, h),
                || {
                    let vc = CellVarContrast::new(rgba, w, h);
                    vc.z_curve()
                },
            )
        },
        || {
            let (ch, nc) = reconsearch::prep(rgba, w, h);
            let build = |y_axis: bool, extent: usize| {
                let ad = reconsearch::AxisData::new(&ch, w, h, nc, y_axis);
                let full = reconsearch::s_grid(extent);
                let s_list: Vec<f64> = full.iter().cloned().step_by(3).collect();
                let (eb, _er) = reconsearch::coarse_curves(&ad, &s_list);
                let trend = reconsearch::Trend::new(&s_list, &eb);
                (ad, trend)
            };
            let (rx, ry) = rayon::join(|| build(false, w), || build(true, h));
            (rx, ry)
        },
    );
    let (vc_logs, vc_z) = vc_data;
    let ((ad_x, trend_x), (ad_y, trend_y)) = recon;

    let steps = fusionchan::ladder();
    let log_steps: Vec<f64> = steps.iter().map(|&s| s.ln()).collect();
    let norm_curve = |c: Vec<f64>| -> Vec<f64> {
        let m = c.iter().cloned().fold(f64::MIN, f64::max);
        if m > 0.0 {
            c.iter().map(|&v| v / m).collect()
        } else {
            c
        }
    };
    let (curve_x, curve_y) = rayon::join(
        || norm_curve(fusionchan::fused_curve(&ev.x, &steps)),
        || norm_curve(fusionchan::fused_curve(&ev.y, &steps)),
    );

    // fused-argmax fourth voter
    let argmax_first = |c: &[f64]| -> usize {
        let mut i = 0;
        for j in 0..c.len() {
            if c[j] > c[i] {
                i = j;
            }
        }
        i
    };
    let fu_sx = steps[argmax_first(&curve_x)];
    let fu_sy = steps[argmax_first(&curve_y)];
    let p_fu = Prop {
        step_x: fu_sx,
        step_y: fu_sy,
        cols: std::cmp::max(1, (w as f64 / fu_sx).round_ties_even() as i64),
        rows: std::cmp::max(1, (h as f64 / fu_sy).round_ties_even() as i64),
    };

    // ---- stage 1: consensus on output size (supermajority >= 3 of 4)
    let named: [(&str, &Prop); 4] = [("ac", &p_ac), ("rl", &p_rl), ("ss", &p_ss), ("fu", &p_fu)];
    {
        let mut groups: Vec<(usize, usize, Vec<usize>)> = Vec::new();
        for (i1, (_, p1)) in named.iter().enumerate() {
            let grp: Vec<usize> = named
                .iter()
                .enumerate()
                .filter(|(_, (_, p2))| size_close(p2, p1))
                .map(|(i2, _)| i2)
                .collect();
            groups.push((grp.len(), i1, grp));
        }
        groups.sort_by(|a, b| b.0.cmp(&a.0)); // stable: ties keep order
        let groups: Vec<_> = groups
            .into_iter()
            .filter(|(_, i1, _)| aspect_ok(named[*i1].1))
            .collect();
        if let Some((cnt, _, grp)) = groups.first() {
            if *cnt >= 3 {
                let cols = int_median(&grp.iter().map(|&i| named[i].1.cols).collect::<Vec<_>>());
                let rows = int_median(&grp.iter().map(|&i| named[i].1.rows).collect::<Vec<_>>());
                let names: Vec<&str> = grp.iter().map(|&i| named[i].0).collect();
                return CoreResult {
                    step_x: w as f64 / cols as f64,
                    step_y: h as f64 / rows as f64,
                    cols,
                    rows,
                    consensus: names.join("+"),
                };
            }
        }
    }

    // ---- stage 2: arbitration
    let detail_cap_x = 8.0 * acf_width(&pre.ex.acf);
    let detail_cap_y = 8.0 * acf_width(&pre.ey.acf);

    let recon_at = |x_axis: bool, s: f64| -> f64 {
        let (ad, trend, extent) = if x_axis {
            (&ad_x, &trend_x, w)
        } else {
            (&ad_y, &trend_y, h)
        };
        if s < 2.0 || s > extent as f64 / 8.0 {
            return 0.0;
        }
        let (eb1, er1) = ad.eval_s(s, false);
        reconsearch::score(eb1, er1, trend.at(s))
    };
    let fused_at = |x_axis: bool, s: f64| -> f64 {
        if s < steps[0] || s > steps[steps.len() - 1] {
            return 0.0;
        }
        let curve = if x_axis { &curve_x } else { &curve_y };
        crate::sigproc::interp(s.ln(), &log_steps, curve)
    };
    let vc_at = |s: f64| -> f64 {
        (vc_z_of(&vc_logs, &vc_z, s) / 10.0)
            .clamp(0.0, 1.0)
            .max(0.0)
    };

    let pick_axis = |x_axis: bool, ac_cands: &[(f64, f64)], extent: usize| -> f64 {
        let sources: Vec<f64> = named
            .iter()
            .map(|(_, p)| if x_axis { p.step_x } else { p.step_y })
            .collect();
        let mut pool: Vec<f64> = sources.clone();
        pool.extend(ac_cands.iter().take(5).map(|c| c.0));

        let mut cands: Vec<f64> = Vec::new();
        for &s in &pool {
            if s <= 1.2 || s > extent as f64 / 3.0 {
                continue;
            }
            if cands.iter().any(|&s2| (s / s2).ln().abs() < 0.01) {
                continue;
            }
            cands.push(s);
        }
        if cands.is_empty() {
            return if x_axis { p_ac.step_x } else { p_ac.step_y };
        }
        let rn: Vec<f64> = cands.iter().map(|&s| recon_at(x_axis, s)).collect();
        let rmax = rn.iter().cloned().fold(f64::MIN, f64::max);
        let recon_ok = rmax > 0.005;
        let cap = if x_axis { detail_cap_x } else { detail_cap_y };
        let mut scored: Vec<(f64, f64)> = Vec::new();
        for (i, &s) in cands.iter().enumerate() {
            let agree = sources
                .iter()
                .filter(|&&v| ((v.max(1e-9)) / s).ln().abs() < AGREE_TOL)
                .count();
            let mut sc =
                fused_at(x_axis, s) + VC_W * vc_at(s) + AGREE_BONUS * (agree as f64 - 1.0).max(0.0);
            if recon_ok {
                sc += 0.6 * (rn[i] / rmax);
            }
            if s > cap {
                sc *= 0.35;
            }
            scored.push((s, sc));
        }
        let best = scored.iter().map(|&(_, sc)| sc).fold(f64::MIN, f64::max);
        scored
            .iter()
            .filter(|&&(_, sc)| sc >= SMALLEST_QUALIFIED * best)
            .map(|&(s, _)| s)
            .fold(f64::INFINITY, f64::min)
    };

    let (mut sx, mut sy) = rayon::join(
        || pick_axis(true, &pre.ex.cands, w),
        || pick_axis(false, &pre.ey.cands, h),
    );

    // aspect guard: prefer the finer step when it has real fused support
    if (sx / sy).ln().abs() > 0.45 {
        let (s_fine, ax_fine_x) = if sx < sy { (sx, true) } else { (sy, false) };
        if fused_at(ax_fine_x, s_fine) >= 0.35 {
            sx = s_fine;
            sy = s_fine;
        }
    }
    if (sx / sy).ln().abs() > 0.45 {
        let both = |s: f64| -> f64 {
            let rx = recon_at(true, s);
            let ry = recon_at(false, s);
            let r_term = if rx.max(ry) > 0.005 {
                0.5 * (rx + ry) / rx.max(ry).max(1e-9)
            } else {
                0.0
            };
            let mut v = r_term + fused_at(true, s) + fused_at(false, s) + 2.0 * VC_W * vc_at(s);
            if s > detail_cap_x.min(detail_cap_y) {
                v *= 0.35;
            }
            v
        };
        if both(sx) >= both(sy) {
            sy = sx;
        } else {
            sx = sy;
        }
    }

    sx = crate::acf::refine_step_acf(&pre.ex.acf, sx, 24);
    sy = crate::acf::refine_step_acf(&pre.ey.acf, sy, 24);

    if (sx / sy).ln().abs() < 0.08 && (sx - sy).abs() > 1e-6 {
        let s = 2.0 * sx * sy / (sx + sy);
        sx = crate::acf::refine_step_acf(&pre.ex.acf, s, 24);
        sy = crate::acf::refine_step_acf(&pre.ey.acf, s, 24);
    }

    let (n_cols, n_rows) = rayon::join(
        || autocorr::local_count(&pre.maps_x, w, sx, &mut rustfft::FftPlanner::new()),
        || autocorr::local_count(&pre.maps_y, h, sy, &mut rustfft::FftPlanner::new()),
    );

    CoreResult {
        step_x: w as f64 / n_cols,
        step_y: h as f64 / n_rows,
        cols: n_cols.round_ties_even().max(1.0) as i64,
        rows: n_rows.round_ties_even().max(1.0) as i64,
        consensus: "arbitrated".into(),
    }
}
