// VDV2 wire format — port of src/format.ts.
// Platform-agnostic: works on native (cargo test) and WASM.
//
// Layout (all little-endian, f32/u32):
//   [ Header 64B ][ X pointCount*4 ][ Y seriesCount*pointCount*4 ][ Meta seriesCount*48 ][ Axes axisCount*32 ]
//
// Header (16 × u32/f32 = 64 bytes):
//   0  magic        u32  'VDV2' (0x32564456)
//   1  version      u32  2
//   2  flags        u32  bit 0 = sharedX
//   3  seriesCount  u32
//   4  pointCount   u32
//   5  xOffset      u32  byte offset of X array
//   6  yOffset      u32  byte offset of Y matrix
//   7  metaOffset   u32  byte offset of per-series meta
//   8  xMin         f32
//   9  xMax         f32
//  10  yMin         f32
//  11  yMax         f32
//  12  axisCount    u32
//  13  axisOffset   u32  byte offset of per-axis config
//  14-15 reserved

pub const MAGIC: u32 = 0x32564456;
pub const VERSION: u32 = 2;
pub const HEADER_BYTES: usize = 64;
pub const META_STRIDE: usize = 48; // 12 f32 per series
pub const AXIS_STRIDE: usize = 32; // 8 f32 per axis
pub const FLAG_SHARED_X: u32 = 1;

/// Allocate a zeroed VDV2 buffer and write the header + default meta/axes.
/// `groups[a]` = number of series on axis `a`.
pub fn create_series_buffer(groups: &[u32], point_count: u32) -> Vec<u8> {
    assert!(!groups.is_empty() && groups.iter().all(|&g| g > 0));
    assert!(point_count > 0);

    let series_count = groups.iter().sum::<u32>() as usize;
    let axis_count = groups.len();
    let x_bytes = point_count as usize * 4;
    let y_bytes = series_count * point_count as usize * 4;
    let meta_bytes = series_count * META_STRIDE;
    let axes_bytes = axis_count * AXIS_STRIDE;
    let total = HEADER_BYTES + x_bytes + y_bytes + meta_bytes + axes_bytes;

    let mut buf = vec![0u8; total];

    let x_off = HEADER_BYTES as u32;
    let y_off = (HEADER_BYTES + x_bytes) as u32;
    let meta_off = (HEADER_BYTES + x_bytes + y_bytes) as u32;
    let axes_off = (HEADER_BYTES + x_bytes + y_bytes + meta_bytes) as u32;

    {
        let h = hdr_u32_mut(&mut buf);
        h[0] = MAGIC;
        h[1] = VERSION;
        h[2] = FLAG_SHARED_X;
        h[3] = series_count as u32;
        h[4] = point_count;
        h[5] = x_off;
        h[6] = y_off;
        h[7] = meta_off;
        h[12] = axis_count as u32;
        h[13] = axes_off;
    }

    // Default meta: white, 2.5 px, axis from groups, no points.
    {
        let start = meta_off as usize;
        let end = start + meta_bytes;
        let meta: &mut [f32] = bytemuck::cast_slice_mut(&mut buf[start..end]);
        let mut s = 0usize;
        for (a, &g) in groups.iter().enumerate() {
            for _ in 0..g {
                let o = s * 12;
                meta[o] = 1.0; meta[o + 1] = 1.0; meta[o + 2] = 1.0; meta[o + 3] = 1.0;
                meta[o + 4] = 2.5;        // lineWidth
                meta[o + 5] = a as f32;   // axisIndex
                meta[o + 6] = 0.0;        // pointSize
                meta[o + 7] = 0.0;        // pointShape
                meta[o + 8] = 1.0; meta[o + 9] = 1.0; meta[o + 10] = 1.0; meta[o + 11] = 1.0;
                s += 1;
            }
        }
    }

    // Default axes: white.
    {
        let start = axes_off as usize;
        let end = start + axes_bytes;
        let axes: &mut [f32] = bytemuck::cast_slice_mut(&mut buf[start..end]);
        for a in 0..axis_count {
            let o = a * 8;
            axes[o] = 1.0; axes[o + 1] = 1.0; axes[o + 2] = 1.0; axes[o + 3] = 1.0;
        }
    }

    buf
}

// ---- raw accessors (all read from the VDV2 &[u8] slice) ----------------------

pub fn hdr_u32(buf: &[u8]) -> &[u32] {
    bytemuck::cast_slice(&buf[..HEADER_BYTES])
}
pub fn hdr_f32(buf: &[u8]) -> &[f32] {
    bytemuck::cast_slice(&buf[..HEADER_BYTES])
}
pub fn hdr_u32_mut(buf: &mut [u8]) -> &mut [u32] {
    bytemuck::cast_slice_mut(&mut buf[..HEADER_BYTES])
}
pub fn hdr_f32_mut(buf: &mut [u8]) -> &mut [f32] {
    bytemuck::cast_slice_mut(&mut buf[..HEADER_BYTES])
}

pub fn series_count(buf: &[u8]) -> usize { hdr_u32(buf)[3] as usize }
pub fn point_count(buf: &[u8]) -> usize  { hdr_u32(buf)[4] as usize }
pub fn axis_count(buf: &[u8]) -> usize   { hdr_u32(buf)[12] as usize }

pub fn x_bytes(buf: &[u8]) -> &[u8] {
    let h = hdr_u32(buf);
    let off = h[5] as usize;
    let len = h[4] as usize * 4;
    &buf[off..off + len]
}
pub fn y_bytes(buf: &[u8]) -> &[u8] {
    let h = hdr_u32(buf);
    let off = h[6] as usize;
    let len = h[3] as usize * h[4] as usize * 4;
    &buf[off..off + len]
}
pub fn meta_bytes(buf: &[u8]) -> &[u8] {
    let h = hdr_u32(buf);
    let off = h[7] as usize;
    let len = h[3] as usize * META_STRIDE;
    &buf[off..off + len]
}
pub fn axes_bytes(buf: &[u8]) -> &[u8] {
    let h = hdr_u32(buf);
    let off = h[13] as usize;
    let len = h[12] as usize * AXIS_STRIDE;
    &buf[off..off + len]
}

pub fn x_slice(buf: &[u8]) -> &[f32]    { bytemuck::cast_slice(x_bytes(buf)) }
pub fn y_slice(buf: &[u8]) -> &[f32]    { bytemuck::cast_slice(y_bytes(buf)) }
pub fn meta_slice(buf: &[u8]) -> &[f32] { bytemuck::cast_slice(meta_bytes(buf)) }
pub fn axes_slice(buf: &[u8]) -> &[f32] { bytemuck::cast_slice(axes_bytes(buf)) }

pub fn get_series_axis(buf: &[u8], series: usize) -> usize {
    meta_slice(buf)[series * 12 + 5].round() as usize
}

pub fn get_axis_color(buf: &[u8], axis: usize) -> [f32; 4] {
    let a = axes_slice(buf);
    let o = axis * 8;
    [a[o], a[o + 1], a[o + 2], a[o + 3]]
}

pub fn recompute_extents(buf: &mut [u8]) {
    let x_min = x_slice(buf).iter().cloned().fold(f32::INFINITY, f32::min);
    let x_max = x_slice(buf).iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let y_min = y_slice(buf).iter().cloned().fold(f32::INFINITY, f32::min);
    let y_max = y_slice(buf).iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let h = hdr_f32_mut(buf);
    h[8] = x_min; h[9] = x_max; h[10] = y_min; h[11] = y_max;
}

pub fn get_extents(buf: &[u8]) -> (f32, f32, f32, f32) {
    let h = hdr_f32(buf);
    (h[8], h[9], h[10], h[11])
}

/// CPU per-axis Y extents (used for initial view before GPU readback).
pub fn get_axis_cpu_extents(buf: &[u8]) -> Vec<(f32, f32)> {
    let sc = series_count(buf);
    let pc = point_count(buf);
    let ac = axis_count(buf);
    let y = y_slice(buf);
    let mut out = vec![(f32::INFINITY, f32::NEG_INFINITY); ac];
    for s in 0..sc {
        let a = get_series_axis(buf, s);
        let base = s * pc;
        for i in 0..pc {
            let v = y[base + i];
            if v < out[a].0 { out[a].0 = v; }
            if v > out[a].1 { out[a].1 = v; }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_layout() {
        let buf = create_series_buffer(&[3, 2, 2], 128);
        let h = hdr_u32(&buf);
        assert_eq!(h[0], MAGIC);
        assert_eq!(h[1], VERSION);
        assert_eq!(h[3], 7); // 3+2+2
        assert_eq!(h[4], 128);
        assert_eq!(h[5] as usize, HEADER_BYTES);
        assert_eq!(h[6] as usize, HEADER_BYTES + 128 * 4);
        assert_eq!(h[7] as usize, HEADER_BYTES + 128 * 4 + 7 * 128 * 4);
        assert_eq!(h[12], 3);
        assert_eq!(h[13] as usize, HEADER_BYTES + 128 * 4 + 7 * 128 * 4 + 7 * META_STRIDE);
    }

    #[test]
    fn default_meta_values() {
        let buf = create_series_buffer(&[1], 4);
        let m = meta_slice(&buf);
        assert_eq!(m[0..4], [1.0, 1.0, 1.0, 1.0]); // white line
        assert_eq!(m[4], 2.5);  // lineWidth
        assert_eq!(m[5], 0.0);  // axisIndex 0
        assert_eq!(m[6], 0.0);  // pointSize off
    }

    #[test]
    fn axis_assignment() {
        let buf = create_series_buffer(&[2, 3], 4);
        assert_eq!(get_series_axis(&buf, 0), 0);
        assert_eq!(get_series_axis(&buf, 1), 0);
        assert_eq!(get_series_axis(&buf, 2), 1);
        assert_eq!(get_series_axis(&buf, 4), 1);
    }

    #[test]
    fn extents_roundtrip() {
        let mut buf = create_series_buffer(&[1], 4);
        let x_off = hdr_u32(&buf)[5] as usize;
        let y_off = hdr_u32(&buf)[6] as usize;
        {
            let xs: &mut [f32] = bytemuck::cast_slice_mut(&mut buf[x_off..x_off + 16]);
            xs.copy_from_slice(&[0.0, 1.0, 2.0, 3.0]);
        }
        {
            let ys: &mut [f32] = bytemuck::cast_slice_mut(&mut buf[y_off..y_off + 16]);
            ys.copy_from_slice(&[-5.0, 3.0, 1.0, 7.0]);
        }
        recompute_extents(&mut buf);
        let (xmin, xmax, ymin, ymax) = get_extents(&buf);
        assert_eq!(xmin, 0.0);
        assert_eq!(xmax, 3.0);
        assert_eq!(ymin, -5.0);
        assert_eq!(ymax, 7.0);
    }

    #[test]
    fn axis_cpu_extents() {
        let mut buf = create_series_buffer(&[1, 1], 4);
        let y_off = hdr_u32(&buf)[6] as usize;
        {
            let ys: &mut [f32] = bytemuck::cast_slice_mut(&mut buf[y_off..y_off + 32]);
            // series 0 (axis 0): 1,2,3,4
            ys[0] = 1.0; ys[1] = 2.0; ys[2] = 3.0; ys[3] = 4.0;
            // series 1 (axis 1): -10,-5,0,5
            ys[4] = -10.0; ys[5] = -5.0; ys[6] = 0.0; ys[7] = 5.0;
        }
        let exts = get_axis_cpu_extents(&buf);
        assert_eq!(exts.len(), 2);
        assert_eq!(exts[0], (1.0, 4.0));
        assert_eq!(exts[1], (-10.0, 5.0));
    }

    #[test]
    fn buffer_size_matches_calculation() {
        let groups = [3u32, 2, 2];
        let pc = 64usize;
        let sc = 7usize;
        let ac = 3usize;
        let expected = HEADER_BYTES + pc * 4 + sc * pc * 4 + sc * META_STRIDE + ac * AXIS_STRIDE;
        let buf = create_series_buffer(&groups, pc as u32);
        assert_eq!(buf.len(), expected);
    }

    #[test]
    fn x_slice_write_read() {
        let mut buf = create_series_buffer(&[1], 4);
        let x_off = hdr_u32(&buf)[5] as usize;
        {
            let xs: &mut [f32] = bytemuck::cast_slice_mut(&mut buf[x_off..x_off + 16]);
            xs.copy_from_slice(&[0.0, 0.5, 0.75, 1.0]);
        }
        let xs = x_slice(&buf);
        assert_eq!(xs, &[0.0f32, 0.5, 0.75, 1.0]);
    }

    #[test]
    fn y_slice_write_read() {
        let mut buf = create_series_buffer(&[2], 3); // 2 series, 3 points
        let y_off = hdr_u32(&buf)[6] as usize;
        {
            let ys: &mut [f32] = bytemuck::cast_slice_mut(&mut buf[y_off..y_off + 24]);
            // series 0: 1,2,3 · series 1: 4,5,6
            ys.copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        }
        let ys = y_slice(&buf);
        assert_eq!(ys[0..3], [1.0f32, 2.0, 3.0]);
        assert_eq!(ys[3..6], [4.0f32, 5.0, 6.0]);
    }

    #[test]
    fn get_axis_color_default() {
        let buf = create_series_buffer(&[2, 1], 4);
        assert_eq!(get_axis_color(&buf, 0), [1.0, 1.0, 1.0, 1.0]);
        assert_eq!(get_axis_color(&buf, 1), [1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn multi_series_same_axis_cpu_extents() {
        // 3 series all on axis 0
        let mut buf = create_series_buffer(&[3], 2);
        let y_off = hdr_u32(&buf)[6] as usize;
        {
            let ys: &mut [f32] = bytemuck::cast_slice_mut(&mut buf[y_off..y_off + 24]);
            ys.copy_from_slice(&[10.0, 20.0, -5.0, 30.0, 7.0, 3.0]);
        }
        let exts = get_axis_cpu_extents(&buf);
        assert_eq!(exts.len(), 1);
        assert_eq!(exts[0].0, -5.0);
        assert_eq!(exts[0].1, 30.0);
    }

    #[test]
    fn accessors_return_correct_counts() {
        let buf = create_series_buffer(&[2, 3], 16);
        assert_eq!(series_count(&buf), 5);
        assert_eq!(point_count(&buf), 16);
        assert_eq!(axis_count(&buf), 2);
    }
}
