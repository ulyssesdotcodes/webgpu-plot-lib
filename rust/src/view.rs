// View math — port of the inline functions in src/chart.ts.

/// Compute X scale/offset so the data range maps to the plot sub-rectangle.
/// Returns (scale_x, offset_x) for the View uniform.
pub fn compute_x_scale_offset(
    data_min_x: f32,
    data_max_x: f32,
    css_width: f32,
    gutter_left: f32,
    gutter_right: f32,
) -> (f32, f32) {
    let plot_w = (css_width - gutter_left - gutter_right).max(1.0);
    let range = (data_max_x - data_min_x).max(1e-30);
    let scale_x = (plot_w / css_width) * 2.0 / range;
    let centre = (data_min_x + data_max_x) * 0.5;
    let plot_centre_ndc = 2.0 * (gutter_left + plot_w * 0.5) / css_width - 1.0;
    let offset_x = centre - plot_centre_ndc / scale_x;
    (scale_x, offset_x)
}

/// "Nice" step for tick spacing.
pub fn nice_step(range: f32, viewport_px: f32, target_px: f32) -> f32 {
    if range <= 0.0 || viewport_px <= 0.0 { return 1.0; }
    let target_count = (viewport_px / target_px).max(2.0);
    let raw = range / target_count;
    let exp = raw.log10().floor();
    let base = raw / 10_f32.powf(exp);
    let nice: f32 = if base < 1.5 { 1.0 } else if base < 3.5 { 2.0 } else if base < 7.5 { 5.0 } else { 10.0 };
    nice * 10_f32.powf(exp)
}

/// Format a tick label (same decimals as the TS version).
pub fn format_tick(v: f32, step: f32) -> String {
    let decimals = if step <= 0.0 { 0 } else { (-step.log10().floor()).max(0.0) as usize };
    format!("{:.prec$}", v, prec = decimals)
}

/// Binary-search the visible point range [first, last] inclusive.
pub fn visible_points(x: &[f32], data_min_x: f32, data_max_x: f32) -> (usize, usize) {
    let n = x.len();
    if n == 0 { return (0, 0); }
    let first = {
        let (mut lo, mut hi) = (0, n);
        while lo < hi { let m = (lo + hi) / 2; if x[m] < data_min_x { lo = m + 1 } else { hi = m } }
        lo.saturating_sub(1)
    };
    let last = {
        let (mut lo, mut hi) = (0, n);
        while lo < hi { let m = (lo + hi) / 2; if x[m] <= data_max_x { lo = m + 1 } else { hi = m } }
        lo.min(n - 1)
    };
    (first, last)
}

/// Visible spline-sample range (in SUBDIVS-expanded space).
pub fn visible_samples(x: &[f32], data_min_x: f32, data_max_x: f32, subdivs: usize) -> (usize, usize) {
    let n = x.len();
    if n < 2 { return (0, 0); }
    let tot = subdivs * (n - 1) + 1;
    let first_pt = {
        let (mut lo, mut hi) = (0, n);
        while lo < hi { let m = (lo + hi) / 2; if x[m] < data_min_x { lo = m + 1 } else { hi = m } }
        lo
    };
    let last_pt = {
        let (mut lo, mut hi) = (0, n);
        while lo < hi { let m = (lo + hi) / 2; if x[m] <= data_max_x { lo = m + 1 } else { hi = m } }
        lo.saturating_sub(1)
    };
    let first = (first_pt * subdivs).saturating_sub(subdivs);
    let last = ((last_pt * subdivs) + subdivs).min(tot - 1);
    (first, last)
}

/// Binary-search the nearest point index to a given data-X.
pub fn pick_x(x: &[f32], data_x: f32) -> usize {
    let n = x.len();
    if n == 0 { return 0; }
    let (mut lo, mut hi) = (0usize, n);
    while lo < hi { let m = (lo + hi) / 2; if x[m] < data_x { lo = m + 1 } else { hi = m } }
    if lo > 0 && lo < n && (x[lo - 1] - data_x).abs() <= (x[lo] - data_x).abs() {
        lo - 1
    } else {
        lo.min(n - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nice_step_basics() {
        // range=10, viewport=800, target=100 → targetCount=8, raw=1.25 → nice=1
        assert!((nice_step(10.0, 800.0, 100.0) - 1.0).abs() < 1e-5);
        // range=100, viewport=800, target=100 → targetCount=8, raw=12.5 → nice=10
        assert!((nice_step(100.0, 800.0, 100.0) - 10.0).abs() < 1e-4);
    }

    #[test]
    fn visible_points_basic() {
        let x = vec![0.0f32, 1.0, 2.0, 3.0, 4.0];
        let (first, last) = visible_points(&x, 1.0, 3.0);
        assert!(first <= 1 && last >= 3);
    }

    #[test]
    fn pick_x_nearest() {
        let x = vec![0.0f32, 1.0, 2.0, 3.0];
        assert_eq!(pick_x(&x, 1.4), 1);
        assert_eq!(pick_x(&x, 1.6), 2);
    }
}
