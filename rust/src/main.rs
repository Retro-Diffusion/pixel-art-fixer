use pixelfixer::{autocorr, core, reconstruct, runlengths, selfsim};

#[cfg(target_os = "windows")]
/// Peak working-set size (bytes) of this process, via psapi.
fn peak_rss() -> u64 {
    #[repr(C)]
    struct Pmc {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentProcess() -> isize;
        fn K32GetProcessMemoryInfo(h: isize, pmc: *mut Pmc, cb: u32) -> i32;
    }
    unsafe {
        let mut pmc: Pmc = std::mem::zeroed();
        pmc.cb = std::mem::size_of::<Pmc>() as u32;
        if K32GetProcessMemoryInfo(GetCurrentProcess(), &mut pmc, pmc.cb) != 0 {
            pmc.peak_working_set_size as u64
        } else {
            0
        }
    }
}

#[cfg(not(target_os = "windows"))]
/// Peak working-set size (bytes) of this process, via getrusage.
fn peak_rss() -> u64 {
    let mut rusage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    let maxrss_kb = unsafe {
        libc::getrusage(libc::RUSAGE_SELF, rusage.as_mut_ptr());
        rusage.assume_init()
    }
    .ru_maxrss as u64;
    maxrss_kb * 1024
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: pixelfixer <command> ...");
        eprintln!("  process    <image> <out.png> [full|fast]  detect + reconstruct (default full)");
        eprintln!("  fast       <image...>                   fast consensus detection, JSON per line");
        eprintln!("  recon      <image> <sx> <sy> <cols> <rows> <out.png> [dark]");
        eprintln!("  autocorr   <image...>                   single-channel detection (debug)");
        eprintln!("  runlengths <image...>                   single-channel detection (debug)");
        eprintln!("  selfsim    <image...>                   single-channel detection (debug)");
        std::process::exit(2);
    }
    let cmd = args[1].as_str();
    match cmd {
        "autocorr" => {
            for path in &args[2..] {
                let img = match image::open(path) {
                    Ok(i) => i.to_rgba8(),
                    Err(e) => {
                        eprintln!("{{\"file\": {:?}, \"error\": {:?}}}", path, e.to_string());
                        continue;
                    }
                };
                let (w, h) = (img.width() as usize, img.height() as usize);
                let t0 = std::time::Instant::now();
                let d = autocorr::detect(img.as_raw(), w, h);
                let dt = t0.elapsed().as_secs_f64();
                println!(
                    "{{\"file\": {:?}, \"step_x\": {:.6}, \"step_y\": {:.6}, \"cols\": {}, \"rows\": {}, \"secs\": {:.3}}}",
                    path, d.step_x, d.step_y, d.cols, d.rows, dt
                );
            }
        }
        "process" => {
            // process <image> <out.png> [full|fast] : detect + reconstruct.
            // Full mode is the default (production setting).
            let path = &args[2];
            let out_path = &args[3];
            let mode = args.get(4).map(|s| s.as_str()).unwrap_or("full");
            let fail = |msg: &str| -> ! {
                eprintln!("{{\"file\": {:?}, \"error\": {:?}}}", path, msg);
                std::process::exit(1);
            };
            let img = match image::open(path) {
                Ok(i) => i.to_rgba8(),
                Err(e) => fail(&format!("cannot open image: {}", e)),
            };
            let (w, h) = (img.width() as usize, img.height() as usize);
            // server input caps (mirror detector/api.py)
            if w.min(h) < 16 {
                fail("image too small (min side 16px)");
            }
            if w * h > 4_000_000 {
                fail(&format!("image too large ({:.1}MP > 4MP limit)",
                              (w * h) as f64 / 1e6));
            }
            let t0 = std::time::Instant::now();
            let d = if mode == "fast" {
                core::detect_fast(img.as_raw(), w, h)
            } else {
                core::detect_full(img.as_raw(), w, h)
            };
            let detect_s = t0.elapsed().as_secs_f64();
            let t0 = std::time::Instant::now();
            // default: two-stage packing (quantise-for-structure +
            // original-pixels-for-colour, adaptive K). Pass "legacy" for the
            // old mode-vote reconstruction.
            let r = if args.iter().any(|s| s == "legacy") {
                reconstruct::reconstruct(
                    img.as_raw(), w, h, d.step_x, d.step_y,
                    d.cols as usize, d.rows as usize, false, false,
                )
            } else {
                reconstruct::two_stage_pack(
                    img.as_raw(), w, h, d.cols as usize, d.rows as usize, 0,
                )
            };
            let recon_s = t0.elapsed().as_secs_f64();
            let buf = match image::RgbaImage::from_raw(r.cols as u32, r.rows as u32, r.rgba) {
                Some(b) => b,
                None => fail("internal: reconstruction buffer size mismatch"),
            };
            if let Err(e) = buf.save(out_path) {
                fail(&format!("cannot write output: {}", e));
            }
            println!(
                "{{\"file\": {:?}, \"cols\": {}, \"rows\": {}, \"step_x\": {:.4}, \"step_y\": {:.4}, \"consensus\": {:?}, \"mode\": {:?}, \"detect_s\": {:.3}, \"recon_s\": {:.3}, \"peak_rss_mb\": {:.1}}}",
                path, r.cols, r.rows, d.step_x, d.step_y, d.consensus, mode,
                detect_s, recon_s, peak_rss() as f64 / 1048576.0
            );
        }
        "recon" => {
            // recon <image> <step_x> <step_y> <cols> <rows> <out.png> [dark]
            let path = &args[2];
            let sx: f64 = args[3].parse().unwrap();
            let sy: f64 = args[4].parse().unwrap();
            let cols: usize = args[5].parse().unwrap();
            let rows: usize = args[6].parse().unwrap();
            let out_path = &args[7];
            let dark = args.get(8).map(|s| s == "dark").unwrap_or(false);
            // low-level primitive: raw (un-quantized) by default so the
            // verify harness stays exact; pass "palette" to enable auto_palette
            let auto_palette = args.iter().any(|s| s == "palette");
            let img = image::open(path).unwrap().to_rgba8();
            let (w, h) = (img.width() as usize, img.height() as usize);
            let t0 = std::time::Instant::now();
            let r = reconstruct::reconstruct(img.as_raw(), w, h, sx, sy, cols, rows, dark, auto_palette);
            let dt = t0.elapsed().as_secs_f64();
            let buf = image::RgbaImage::from_raw(r.cols as u32, r.rows as u32, r.rgba)
                .expect("size");
            buf.save(out_path).unwrap();
            println!(
                "{{\"file\": {:?}, \"cols\": {}, \"rows\": {}, \"secs\": {:.3}}}",
                path, r.cols, r.rows, dt
            );
        }
        "selfsim" => {
            for path in &args[2..] {
                let img = match image::open(path) {
                    Ok(i) => i.to_rgba8(),
                    Err(e) => {
                        eprintln!("{{\"file\": {:?}, \"error\": {:?}}}", path, e.to_string());
                        continue;
                    }
                };
                let (w, h) = (img.width() as usize, img.height() as usize);
                let t0 = std::time::Instant::now();
                let d = selfsim::detect(img.as_raw(), w, h);
                let dt = t0.elapsed().as_secs_f64();
                println!(
                    "{{\"file\": {:?}, \"step_x\": {:.6}, \"step_y\": {:.6}, \"cols\": {}, \"rows\": {}, \"conf\": {:.6}, \"secs\": {:.3}}}",
                    path, d.step_x, d.step_y, d.cols, d.rows, d.conf, dt
                );
            }
        }
        "fast" | "full" => {
            for path in &args[2..] {
                let img = match image::open(path) {
                    Ok(i) => i.to_rgba8(),
                    Err(e) => {
                        eprintln!("{{\"file\": {:?}, \"error\": {:?}}}", path, e.to_string());
                        continue;
                    }
                };
                let (w, h) = (img.width() as usize, img.height() as usize);
                let t0 = std::time::Instant::now();
                let d = if cmd == "full" {
                    core::detect_full(img.as_raw(), w, h)
                } else {
                    core::detect_fast(img.as_raw(), w, h)
                };
                let dt = t0.elapsed().as_secs_f64();
                println!(
                    "{{\"file\": {:?}, \"step_x\": {:.6}, \"step_y\": {:.6}, \"cols\": {}, \"rows\": {}, \"consensus\": {:?}, \"secs\": {:.3}}}",
                    path, d.step_x, d.step_y, d.cols, d.rows, d.consensus, dt
                );
            }
        }
        "runlengths" => {
            for path in &args[2..] {
                let img = match image::open(path) {
                    Ok(i) => i.to_rgba8(),
                    Err(e) => {
                        eprintln!("{{\"file\": {:?}, \"error\": {:?}}}", path, e.to_string());
                        continue;
                    }
                };
                let (w, h) = (img.width() as usize, img.height() as usize);
                let t0 = std::time::Instant::now();
                let d = runlengths::detect(img.as_raw(), w, h);
                let dt = t0.elapsed().as_secs_f64();
                println!(
                    "{{\"file\": {:?}, \"step_x\": {:.6}, \"step_y\": {:.6}, \"cols\": {}, \"rows\": {}, \"score_x\": {:.6}, \"score_y\": {:.6}, \"nruns_x\": {}, \"nruns_y\": {}, \"secs\": {:.3}}}",
                    path, d.step_x, d.step_y, d.cols, d.rows, d.score_x, d.score_y,
                    d.nruns_x, d.nruns_y, dt
                );
            }
        }
        _ => {
            eprintln!("unknown command: {}", cmd);
            std::process::exit(2);
        }
    }
}
