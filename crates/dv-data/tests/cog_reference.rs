//! Reads Cloud Optimized GeoTIFFs, on disk and over HTTP range requests, and checks levels,
//! windows and overview choices against GDAL (rasterio). Made by
//! `scripts/reference/make-cog-fixtures.py`.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use dv_data::array::{self, ArraySource};
use dv_data::raster::GeoTiff;
use serde_json::Value as Json;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

fn reference() -> Json {
    let text = std::fs::read_to_string(fixtures().join("reference/cog.json")).unwrap();
    serde_json::from_str(&text).unwrap()
}

const FILES: [&str; 2] = ["cog-dem.tif", "cog-rgb.tif"];

fn pair(v: &Json) -> [u64; 2] {
    [v[0].as_u64().unwrap(), v[1].as_u64().unwrap()]
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0)
}

#[test]
fn levels_match_gdal() {
    let r = reference();
    for file in FILES {
        let tif = GeoTiff::open(&fixtures().join("geo").join(file)).unwrap();
        let want = r[file]["levels"].as_array().unwrap();
        assert_eq!(tif.levels().len(), want.len(), "{file}: levels");
        for (k, (l, w)) in tif.levels().iter().zip(want).enumerate() {
            assert_eq!(
                [l.width, l.height],
                [w["width"].as_u64().unwrap(), w["height"].as_u64().unwrap()],
                "{file} level {k}: size"
            );
            // GDAL's geotransform: (a, b, c, d, e, f) with x = c + a * col, y = f + e * row.
            let t: Vec<f64> = w["transform"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect();
            for (got, want) in [(l.dx, t[0]), (l.x0, t[2]), (l.dy, t[4]), (l.y0, t[5])] {
                assert!(close(got, want), "{file} level {k}: {got} vs GDAL {want}");
            }
        }
    }
}

/// Checks every reference window of `file`, read through `tif`. Calls `after` with each
/// window's reference after reading it.
fn check_windows(file: &str, tif: &GeoTiff, mut after: impl FnMut(&Json)) {
    let r = reference();
    for w in r[file]["windows"].as_array().unwrap() {
        let level = w["level"].as_u64().unwrap() as usize;
        let bands: Vec<usize> = w["bands"]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| b.as_u64().unwrap() as usize)
            .collect();
        let (start, count) = (pair(&w["start"]), pair(&w["count"]));
        let got = tif.read_bands(level, &bands, start, count).unwrap();
        for (b, (g, want)) in got.iter().zip(w["values"].as_array().unwrap()).enumerate() {
            let want = want.as_array().unwrap();
            assert_eq!(g.len(), want.len());
            for (i, (g, w)) in g.iter().zip(want).enumerate() {
                assert_eq!(
                    *g,
                    w.as_f64().unwrap(),
                    "{file} level {level} band {} {start:?}+{count:?} [{i}]",
                    bands[b]
                );
            }
        }
        after(w);
    }
}

#[test]
fn windows_match_gdal() {
    for file in FILES {
        let tif = GeoTiff::open(&fixtures().join("geo").join(file)).unwrap();
        check_windows(file, &tif, |_| {});
    }
}

#[test]
fn bands_read_through_array_source_match_gdal() {
    // A level 0 window, through `ArraySource`, with nodata masked.
    let r = reference();
    let w = &r["cog-dem.tif"]["windows"][0];
    let src = array::open(fixtures().join("geo/cog-dem.tif")).unwrap();
    let (start, count) = (pair(&w["start"]), pair(&w["count"]));
    let got = array::read(src.as_ref(), "band_1", &start, &count).unwrap();
    let nodata = r["cog-dem.tif"]["nodata"].as_f64().unwrap();
    for (g, w) in got.iter().zip(w["values"][0].as_array().unwrap()) {
        let w = w.as_f64().unwrap();
        assert!(if w == nodata { g.is_nan() } else { *g == w });
    }
    let x = array::read(src.as_ref(), "x", &[start[1]], &[1]).unwrap()[0];
    assert!(close(x, 500000.0 + 30.0 * (start[1] as f64 + 0.5)));
}

#[test]
fn overview_choice_matches_gdal() {
    let r = reference();
    let tif = GeoTiff::open(&fixtures().join("geo/cog-dem.tif")).unwrap();
    for c in r["cog-dem.tif"]["choices"].as_array().unwrap() {
        let (count, out) = (pair(&c["count"]), pair(&c["out"]));
        let got = tif.level_for_shape(count, out) as u64;
        let gdal: Vec<u64> = c["levels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|l| l.as_u64().unwrap())
            .collect();
        assert!(
            gdal.contains(&got),
            "{count:?} into {out:?}: level {got}, GDAL read {gdal:?}"
        );
    }
    // 30 m pixels: 60 m is overview 1, 125 m overview 2 (4.06x, under 1.2 times 125/30), and a
    // resolution finer than the image's is the image.
    assert_eq!(tif.level_for_resolution(60.0), 1);
    assert_eq!(tif.level_for_resolution(125.0), 2);
    assert_eq!(tif.level_for_resolution(10.0), 0);
    assert_eq!(tif.level_for_resolution(1e6), 3);
}

/// A minimal HTTP/1.1 file server for `dir` that answers `Range` requests and logs each GET's
/// byte range.
fn serve(dir: PathBuf) -> (String, Arc<Mutex<Vec<Range<u64>>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let log: Arc<Mutex<Vec<Range<u64>>>> = Arc::default();
    let l = log.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let (dir, log) = (dir.clone(), l.clone());
            std::thread::spawn(move || handle(stream, &dir, &log));
        }
    });
    (url, log)
}

fn handle(stream: TcpStream, dir: &Path, log: &Mutex<Vec<Range<u64>>>) {
    let mut out = stream.try_clone().unwrap();
    let mut reader = BufReader::new(stream);
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).unwrap_or(0) == 0 {
            return;
        }
        let mut parts = line.split_whitespace();
        let (method, path) = (parts.next().unwrap_or(""), parts.next().unwrap_or("/"));
        let mut range = None;
        loop {
            let mut h = String::new();
            reader.read_line(&mut h).unwrap();
            let h = h.trim_end();
            if h.is_empty() {
                break;
            }
            if let Some((k, v)) = h.split_once(':')
                && k.eq_ignore_ascii_case("range")
            {
                let (a, b) = v
                    .trim()
                    .trim_start_matches("bytes=")
                    .split_once('-')
                    .unwrap();
                range = Some((a.parse::<u64>().unwrap(), b.parse::<u64>().ok()));
            }
        }
        let Ok(mut file) = std::fs::File::open(dir.join(path.trim_start_matches('/'))) else {
            write!(out, "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n").unwrap();
            continue;
        };
        let mut body = Vec::new();
        file.read_to_end(&mut body).unwrap();
        let size = body.len() as u64;
        let (status, extra, r) = match range {
            Some((a, b)) => {
                let end = b.map_or(size, |b| (b + 1).min(size));
                let cr = format!("Content-Range: bytes {a}-{}/{size}\r\n", end - 1);
                ("206 Partial Content", cr, a..end)
            }
            None => ("200 OK", String::new(), 0..size),
        };
        let head = method == "HEAD";
        if !head {
            log.lock().unwrap().push(r.clone());
        }
        write!(
            out,
            "HTTP/1.1 {status}\r\nContent-Length: {}\r\n{extra}Accept-Ranges: bytes\r\n\r\n",
            r.end - r.start
        )
        .unwrap();
        if !head {
            out.write_all(&body[r.start as usize..r.end as usize])
                .unwrap();
        }
    }
}

fn fetched(log: &Mutex<Vec<Range<u64>>>) -> u64 {
    log.lock().unwrap().drain(..).map(|r| r.end - r.start).sum()
}

#[test]
fn range_reads_fetch_only_the_tiles_a_window_needs() {
    let (url, log) = serve(fixtures().join("geo"));
    let r = reference();
    for file in FILES {
        let size = r[file]["size"].as_u64().unwrap();
        let tif = GeoTiff::open_url(&format!("{url}/{file}")).unwrap();
        // Opening reads the headers of every level from the first 16 KiB, and nothing more.
        let header = fetched(&log);
        assert!(
            header <= 16 * 1024,
            "{file}: opening fetched {header} bytes"
        );
        assert_eq!(
            tif.levels().len(),
            r[file]["levels"].as_array().unwrap().len()
        );
        // Each read fetches at most the bytes of the tiles it touches (less when the header
        // read already holds some), and gets GDAL's values.
        check_windows(file, &tif, |w| {
            let got = fetched(&log);
            let tiles = w["tile_bytes"].as_u64().unwrap();
            assert!(
                got <= tiles,
                "{file} window {w:?}: fetched {got} bytes for {tiles} bytes of tiles"
            );
            // The DEM's full-resolution tiles follow its overviews, well past the header read,
            // so those windows fetch exactly their tiles.
            if file == "cog-dem.tif" && w["level"] == 0 {
                assert_eq!(got, tiles, "{file} window {w:?}");
            }
        });
        // A small window of the full image costs a sliver of the file.
        let first = tif.levels()[0].clone();
        let v = tif.read_level(0, 1, [first.height - 1, 0], [1, 1]).unwrap();
        assert_eq!(v.len(), 1);
        let got = fetched(&log);
        assert!(
            got > 0 && got < size / 10,
            "{file}: one pixel fetched {got} of {size} bytes"
        );
    }
}

#[test]
fn remote_and_local_reads_agree_through_array_source() {
    let (url, log) = serve(fixtures().join("geo"));
    let local = array::open(fixtures().join("geo/cog-dem.tif")).unwrap();
    let remote = GeoTiff::open_url(&format!("{url}/cog-dem.tif")).unwrap();
    assert_eq!(remote.variables(), local.variables());
    assert_eq!(remote.attrs(), local.attrs());
    for var in ["band_1", "x", "y"] {
        let a = array::read_all(local.as_ref(), var).unwrap();
        let b = array::read_all(&remote, var).unwrap();
        assert_eq!(a.len(), b.len());
        assert!(
            a.iter()
                .zip(&b)
                .all(|(a, b)| a == b || (a.is_nan() && b.is_nan()))
        );
    }
    // A missing file fails to open rather than reading as empty.
    assert!(GeoTiff::open_url(&format!("{url}/missing.tif")).is_err());
    drop(log);
}
