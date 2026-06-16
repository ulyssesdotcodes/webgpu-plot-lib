// wasm-bindgen surface for the Rust/wgpu chart.

pub mod format;
pub mod view;
pub mod labels;

// chart.rs uses wgpu + web-sys and is only compiled for wasm32.
#[cfg(target_arch = "wasm32")]
mod chart;

#[cfg(target_arch = "wasm32")]
use std::rc::Rc;
#[cfg(target_arch = "wasm32")]
use std::cell::RefCell;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use web_sys::HtmlCanvasElement;

#[cfg(target_arch = "wasm32")]
use chart::ChartInner;

/// Opaque chart handle exposed to JavaScript.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct Chart {
    inner: Rc<RefCell<ChartInner>>,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl Chart {
    /// Async factory. `groups` is a Uint32Array of per-axis series counts.
    /// `atlas_meta` is a Float32Array of [uMin, uMax, widthPx] triplets for each
    /// character in TEXT_CHARS order (built once in JS from Canvas-2D).
    /// Returns a Promise<Chart>.
    pub async fn create(
        canvas: HtmlCanvasElement,
        groups: Vec<u32>,
        point_count: u32,
        atlas_rgba: Vec<u8>,
        atlas_width: u32,
        atlas_height: u32,
        atlas_meta: Vec<f32>,
        atlas_h_px: f32,
        bg_r: f32, bg_g: f32, bg_b: f32, bg_a: f32,
        grid_r: f32, grid_g: f32, grid_b: f32, grid_a: f32,
        pixel_ratio: f32,
    ) -> Result<Chart, JsValue> {
        console_error_panic_hook::set_once();

        let inner = ChartInner::create(
            canvas, groups, point_count,
            atlas_rgba, atlas_width, atlas_height, atlas_meta, atlas_h_px,
            [bg_r, bg_g, bg_b, bg_a],
            [grid_r, grid_g, grid_b, grid_a],
            pixel_ratio,
        ).await?;

        Ok(Chart { inner })
    }

    // ---- VDV2 buffer access --------------------------------------------------

    /// Pointer into WASM linear memory where the VDV2 buffer lives.
    /// JS uses this + memory.buffer to build zero-copy typed-array views.
    pub fn buffer_ptr(&self) -> u32 {
        self.inner.borrow().vdv2.as_ptr() as u32
    }

    pub fn buffer_len(&self) -> u32 {
        self.inner.borrow().vdv2.len() as u32
    }

    // ---- data ops ------------------------------------------------------------

    /// Upload the VDV2 buffer (x, y, meta) to GPU buffers and re-bake the spline.
    pub fn upload_data(&self) {
        let mut inner = self.inner.borrow_mut();
        inner.upload_data();
        inner.bake_spline();
    }

    /// Full reset: upload data, recompute extents, reset view, render.
    pub fn set_data(&self) {
        {
            let mut inner = self.inner.borrow_mut();
            inner.upload_data();
            inner.bake_spline();
            inner.reset_view();
            inner.update_axis_extents_cpu();
        }
        ChartInner::request_render(&self.inner);
    }

    // ---- view controls -------------------------------------------------------

    pub fn reset_view(&self) {
        {
            let mut inner = self.inner.borrow_mut();
            inner.upload_data();
            inner.bake_spline();
            inner.reset_view();
            inner.update_axis_extents_cpu();
        }
        ChartInner::request_render(&self.inner);
    }

    pub fn resize(&self, css_w: f32, css_h: f32, dpr: f32) {
        self.inner.borrow_mut().resize(css_w, css_h, dpr);
        ChartInner::request_render(&self.inner);
    }

    pub fn pan(&self, dx_css: f32) {
        self.inner.borrow_mut().pan(dx_css);
        ChartInner::request_render(&self.inner);
    }

    pub fn update_extents(&self) {
        {
            let mut inner = self.inner.borrow_mut();
            inner.update_axis_extents_cpu();
        }
        ChartInner::request_render(&self.inner);
    }

    pub fn zoom_at(&self, css_x: f32, factor: f32) {
        {
            let mut inner = self.inner.borrow_mut();
            inner.zoom_at(css_x, factor);
            inner.update_axis_extents_cpu();
        }
        ChartInner::request_render(&self.inner);
    }

    /// Zoom to a CSS-pixel X range (right-drag selection from JS shim).
    pub fn zoom_to_css_range(&self, css_x0: f32, css_x1: f32) {
        use labels::PX_PER_AXIS;
        let (x0, x1) = {
            let inner = self.inner.borrow();
            let gutter_l = inner.axis_count as f32 * PX_PER_AXIS;
            let plot_w = (inner.css_width - gutter_l).max(1.0);
            let to_data = |cx: f32| -> f32 {
                let fx = ((cx - gutter_l) / plot_w).clamp(0.0, 1.0);
                inner.data_min_x + fx * (inner.data_max_x - inner.data_min_x)
            };
            (to_data(css_x0), to_data(css_x1))
        };
        {
            let mut inner = self.inner.borrow_mut();
            inner.zoom_to_range(x0, x1);
            inner.update_axis_extents_cpu();
        }
        ChartInner::request_render(&self.inner);
    }

    /// Return the nearest data-point index to a CSS-pixel X coordinate.
    pub fn pick_x(&self, css_x: f32) -> i32 {
        self.inner.borrow().pick_x_point(css_x)
    }

    /// Explicit render call (e.g. from a JS requestAnimationFrame loop).
    pub fn render(&self) {
        self.inner.borrow_mut().render();
    }

    pub fn request_render(&self) {
        ChartInner::request_render(&self.inner);
    }
}

// ---- standalone helpers exposed to JS (available on wasm32 only) ----------

/// Recompute x/y global extents in-place in a VDV2 buffer at the given ptr/len.
/// Called from JS after filling x and y data.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn vdv2_recompute_extents(ptr: u32, len: u32) {
    let buf = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u8, len as usize) };
    format::recompute_extents(buf);
}

/// Return the WebAssembly.Memory object so JS can build zero-copy typed-array
/// views over the WASM linear memory at ptr/len offsets from buffer_ptr().
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn wasm_memory() -> wasm_bindgen::JsValue {
    wasm_bindgen::memory()
}
