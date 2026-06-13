// Labels — port of chart.ts updateLabels().
// Builds a flat Vec<f32> of CharInst records (TEXT_INST_FLOATS floats each)
// ready to upload to the textInstBuf.

use crate::view::{nice_step, format_tick};
use crate::format;

/// TEXT_CHARS order determines charMeta index (must match the atlas built in JS).
pub const TEXT_CHARS: &str = "0123456789.-\u{2588}"; // █ at index 12
pub const TEXT_MAX_CHARS: usize = 512;
pub const TEXT_INST_FLOATS: usize = 12;
// floats per CharInst: ndcPos(2) + ndcSize(2) + uvMin(2) + uvMax(2) + color(4)

pub const PX_PER_AXIS: f32 = 55.0;

#[derive(Clone, Copy, Default)]
pub struct CharMeta {
    pub u_min: f32,
    pub u_max: f32,
    pub width: f32,  // px
}

/// Build the flat char-instance array.
/// Returns (instances: Vec<f32>, count: usize).
pub fn build_labels(
    buf: &[u8],
    char_meta: &[CharMeta],      // TEXT_CHARS.len() entries
    atlas_h: f32,                // atlas glyph height in CSS px
    css_w: f32,
    css_h: f32,
    data_min_x: f32,
    data_max_x: f32,
    step_x: f32,                 // current grid step (from view uniform)
    axis_min_y: &[f32],
    axis_max_y: &[f32],
) -> (Vec<f32>, usize) {
    let sc = format::series_count(buf);
    let ac = format::axis_count(buf);

    let mut inst = vec![0.0f32; TEXT_MAX_CHARS * TEXT_INST_FLOATS];
    let mut count = 0usize;

    // Build char → index lookup from TEXT_CHARS.
    let mut char_idx = [usize::MAX; 128];
    let block_char = '\u{2588}';
    let mut block_idx = usize::MAX;
    for (i, c) in TEXT_CHARS.chars().enumerate() {
        if (c as u32) < 128 { char_idx[c as usize] = i; }
        if c == block_char { block_idx = i; }
    }

    let gutter_left  = ac as f32 * PX_PER_AXIS;
    let gutter_right = 0.0f32;
    let plot_w = (css_w - gutter_left - gutter_right).max(1.0);

    let emit_char = |inst: &mut Vec<f32>, count: &mut usize, idx: usize, cur_x: f32, top_y: f32, color: [f32; 4]| -> f32 {
        if *count >= TEXT_MAX_CHARS { return 0.0; }
        let m = char_meta[idx];
        let ndc_x =  (cur_x / css_w) * 2.0 - 1.0;
        let ndc_y = 1.0 - (top_y / css_h) * 2.0;
        let ndc_w = (m.width / css_w) * 2.0;
        let ndc_h = (atlas_h / css_h) * 2.0;
        let off = *count * TEXT_INST_FLOATS;
        inst[off]     = ndc_x;    inst[off + 1]  = ndc_y;
        inst[off + 2] = ndc_w;    inst[off + 3]  = ndc_h;
        inst[off + 4] = m.u_min;  inst[off + 5]  = 0.0;
        inst[off + 6] = m.u_max;  inst[off + 7]  = 1.0;
        inst[off + 8] = color[0]; inst[off + 9]  = color[1];
        inst[off + 10]= color[2]; inst[off + 11] = color[3];
        *count += 1;
        m.width
    };

    let emit_label = |inst: &mut Vec<f32>, count: &mut usize, text: &str, anchor_x: f32, anchor_y: f32, center: bool, color: [f32; 4]| {
        let total_w: f32 = text.chars().filter_map(|c| {
            let i = if (c as u32) < 128 { char_idx[c as usize] } else { usize::MAX };
            if i == usize::MAX { None } else { Some(char_meta[i].width) }
        }).sum();
        let mut cur_x = if center { anchor_x - total_w / 2.0 } else { anchor_x };
        let top_y = anchor_y - atlas_h / 2.0;
        for c in text.chars() {
            let i = if (c as u32) < 128 { char_idx[c as usize] } else { usize::MAX };
            if i == usize::MAX { continue; }
            cur_x += emit_char(inst, count, i, cur_x, top_y, color);
        }
    };

    let white = [1.0f32; 4];

    // X ticks.
    if step_x > 0.0 && plot_w > 0.0 {
        let range_x = data_max_x - data_min_x;
        let first = (data_min_x / step_x).ceil() * step_x;
        let mut v = first;
        while v < data_max_x + step_x * 1e-6 {
            let px = gutter_left + (v - data_min_x) / range_x.max(1e-30) * plot_w;
            if px >= gutter_left && px <= css_w - gutter_right {
                let label = format_tick(v, step_x);
                emit_label(&mut inst, &mut count, &label, px, css_h - atlas_h / 2.0 - 4.0, true, white);
            }
            v += step_x;
        }
    }

    // Y ticks — one column per axis, all on the left.
    for a in 0..ac {
        let y_min = axis_min_y[a];
        let y_max = axis_max_y[a];
        let y_range = y_max - y_min;
        if y_range <= 0.0 { continue; }

        let step_y = nice_step(y_range, css_h, 80.0);
        if step_y <= 0.0 { continue; }

        let col_left = (ac - 1 - a) as f32 * PX_PER_AXIS;
        let label_x = col_left + 4.0;

        let first = (y_min / step_y).ceil() * step_y;
        let mut v = first;
        while v < y_max + step_y * 1e-6 {
            let py = (1.0 - (v - y_min) / y_range) * css_h;
            if py >= 0.0 && py <= css_h {
                let label = format_tick(v, step_y);
                emit_label(&mut inst, &mut count, &label, label_x, py, false, white);
            }
            v += step_y;
        }

        // Colored block (█) per series on this axis.
        let sq_w = if block_idx < char_meta.len() { char_meta[block_idx].width } else { 8.0 };
        let col_center_x = col_left + PX_PER_AXIS / 2.0;
        let series_on_axis: Vec<usize> = (0..sc)
            .filter(|&s| format::get_series_axis(buf, s) == a)
            .collect();
        let mut sq_x = col_center_x - (series_on_axis.len() as f32 * sq_w) / 2.0;
        for s in series_on_axis {
            let m = format::meta_slice(buf);
            let o = s * 12;
            let color = [m[o], m[o + 1], m[o + 2], 1.0];
            if block_idx < char_meta.len() {
                emit_char(&mut inst, &mut count, block_idx, sq_x, 4.0, color);
            }
            sq_x += sq_w;
        }
    }

    (inst, count)
}
