// Main chart — port of src/chart.ts LineChart.
// Only compiled for wasm32; uses wgpu with the WebGPU backend.

use std::rc::Rc;
use std::cell::RefCell;

use wasm_bindgen::JsValue;
use wasm_bindgen::JsCast;
use web_sys::HtmlCanvasElement;

use crate::format;
use crate::labels::{CharMeta, TEXT_INST_FLOATS, TEXT_MAX_CHARS, build_labels};
use crate::view::{compute_x_scale_offset, nice_step, visible_points, visible_samples, pick_x};

// ---- uniform sizes (must match shaders) --------------------------------------
const VIEW_BYTES: u64        = 40;
const STYLE_BYTES: u64       = 32;
const SPLINE_PARAMS_BYTES: u64 = 16;
const AXIS_GPU_BYTES: u64    = 32; // 8 f32 per axis
const SUBDIVS: u32           = 4;
const PX_PER_AXIS: f32       = 55.0;

// Shaders are loaded from the existing public/shaders/ directory at compile time.
const SHADER_GRID:   &str = include_str!("../../public/shaders/grid.wgsl");
const SHADER_LINES:  &str = include_str!("../../public/shaders/lines.wgsl");
const SHADER_POINTS: &str = include_str!("../../public/shaders/points.wgsl");
const SHADER_TEXT:   &str = include_str!("../../public/shaders/text.wgsl");
const SHADER_SPLINE: &str = include_str!("../../public/shaders/spline.wgsl");

pub struct ChartInner {
    // GPU core
    pub surface:        wgpu::Surface<'static>,
    pub device:         wgpu::Device,
    pub queue:          wgpu::Queue,
    pub surface_fmt:    wgpu::TextureFormat,
    pub alpha_mode:     wgpu::CompositeAlphaMode,
    pub surface_w:      u32,
    pub surface_h:      u32,

    // VDV2 data buffer (WASM linear memory)
    pub vdv2:           Vec<u8>,
    pub series_count:   usize,
    pub point_count:    usize,
    pub axis_count:     usize,

    // GPU buffers
    pub view_buf:          wgpu::Buffer,
    pub style_buf:         wgpu::Buffer,
    pub x_buf:             wgpu::Buffer,
    pub y_buf:             wgpu::Buffer,
    pub meta_buf:          wgpu::Buffer,
    pub axes_buf:          wgpu::Buffer,
    pub spline_params_buf: wgpu::Buffer,
    pub spline_buf:        wgpu::Buffer,
    pub text_inst_buf:     wgpu::Buffer,

    // Pipelines
    pub spline_pipeline: wgpu::ComputePipeline,
    pub grid_pipeline:   wgpu::RenderPipeline,
    pub line_pipeline:   wgpu::RenderPipeline,
    pub point_pipeline:  wgpu::RenderPipeline,
    pub text_pipeline:   wgpu::RenderPipeline,

    // Bind groups
    pub spline_bind: wgpu::BindGroup,
    pub grid_bind:   wgpu::BindGroup,
    pub line_bind:   wgpu::BindGroup,
    pub point_bind:  wgpu::BindGroup,
    pub text_bind:   wgpu::BindGroup,

    // Font atlas
    pub atlas_texture: wgpu::Texture,
    pub atlas_sampler: wgpu::Sampler,
    pub char_meta:     Vec<CharMeta>,
    pub atlas_h:       f32,

    // View state
    pub css_width:    f32,
    pub css_height:   f32,
    pub pixel_ratio:  f32,
    pub data_min_x:   f32,
    pub data_max_x:   f32,
    pub axis_min_y:   Vec<f32>,
    pub axis_max_y:   Vec<f32>,

    // Staging for view uniform (10 × f32 / u32, reinterpreted)
    pub view_staging: [u8; VIEW_BYTES as usize],

    // Per-axis GPU staging (8 f32 per axis)
    pub axes_staging: Vec<f32>,

    // Series render lists
    pub line_series:  Vec<usize>,
    pub point_series: Vec<usize>,

    // State flags
    pub view_initialized: bool,
    pub text_char_count:  u32,
    pub render_queued:    bool,

    // Options
    pub background: [f32; 4],
    pub grid_color: [f32; 4],
}

impl ChartInner {
    // ---- public wgpu factory (async, called from lib.rs) ---------------------

    pub async fn create(
        canvas: HtmlCanvasElement,
        groups: Vec<u32>,
        point_count: u32,
        atlas_rgba: Vec<u8>,
        atlas_width: u32,
        atlas_height: u32,
        atlas_meta: Vec<f32>,  // [uMin, uMax, widthPx] × charCount
        atlas_h_px: f32,
        background: [f32; 4],
        grid_color: [f32; 4],
        pixel_ratio: f32,
    ) -> Result<Rc<RefCell<ChartInner>>, JsValue> {
        // GPU init.
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU,
            ..Default::default()
        });

        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
            .map_err(|e| JsValue::from_str(&format!("create_surface: {e}")))?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| JsValue::from_str("no WebGPU adapter"))?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .map_err(|e| JsValue::from_str(&format!("request_device: {e}")))?;

        // Surface configuration.
        let caps = surface.get_capabilities(&adapter);
        let surface_fmt = caps.formats.iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let alpha_mode = caps.alpha_modes[0];

        let canvas_w = canvas.width().max(1);
        let canvas_h = canvas.height().max(1);
        surface.configure(&device, &wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_fmt,
            width: canvas_w,
            height: canvas_h,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        });

        // Allocate VDV2 buffer.
        let vdv2 = format::create_series_buffer(&groups, point_count);
        let series_count = format::series_count(&vdv2);
        let axis_count = format::axis_count(&vdv2);
        let pc = point_count as usize;

        let total_samples = SUBDIVS as usize * (pc - 1) + 1;

        // Build GPU buffers.
        let view_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("view"), size: VIEW_BYTES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });
        let style_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("style"), size: STYLE_BYTES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });
        let x_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("x"), size: (pc * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });
        let y_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("y"), size: (series_count * pc * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });
        let meta_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("meta"), size: (series_count * 48) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });
        let axes_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("axes"), size: (axis_count as u64) * AXIS_GPU_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });
        let spline_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("spline-params"), size: SPLINE_PARAMS_BYTES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });
        let spline_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("spline"), size: (series_count * total_samples * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
        });
        let text_inst_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("text-inst"), size: (TEXT_MAX_CHARS * TEXT_INST_FLOATS * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });

        // Write style uniform (grid color + bg color) once.
        {
            let mut sd = [0f32; 8];
            sd[..4].copy_from_slice(&grid_color);
            sd[4..].copy_from_slice(&background);
            queue.write_buffer(&style_buf, 0, bytemuck::cast_slice(&sd));
        }

        // Shader modules.
        let mk_shader = |code: &str, label: &str| device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label), source: wgpu::ShaderSource::Wgsl(code.into()),
        });
        let grid_mod   = mk_shader(SHADER_GRID,   "grid");
        let lines_mod  = mk_shader(SHADER_LINES,  "lines");
        let points_mod = mk_shader(SHADER_POINTS, "points");
        let text_mod   = mk_shader(SHADER_TEXT,   "text");
        let spline_mod = mk_shader(SHADER_SPLINE, "spline");

        // Compute pipelines.
        let spline_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("spline"), layout: None, module: &spline_mod,
            entry_point: Some("main"), compilation_options: Default::default(),
            cache: None,
        });

        // Render pipelines.
        let grid_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("grid"), layout: None,
            vertex:   wgpu::VertexState { module: &grid_mod, entry_point: Some("vs"), buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: &grid_mod, entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format: surface_fmt, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, ..Default::default() },
            depth_stencil: None, multisample: Default::default(), multiview: None, cache: None,
        });

        let alpha_blend = wgpu::BlendState {
            color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::SrcAlpha, dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha, operation: wgpu::BlendOperation::Add },
            alpha: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One,      dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha, operation: wgpu::BlendOperation::Add },
        };

        let line_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("lines"), layout: None,
            vertex:   wgpu::VertexState { module: &lines_mod, entry_point: Some("vs"), buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: &lines_mod, entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format: surface_fmt, blend: Some(alpha_blend), write_mask: wgpu::ColorWrites::ALL })],
            }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleStrip, ..Default::default() },
            depth_stencil: None, multisample: Default::default(), multiview: None, cache: None,
        });

        let point_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("points"), layout: None,
            vertex:   wgpu::VertexState { module: &points_mod, entry_point: Some("vs"), buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: &points_mod, entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format: surface_fmt, blend: Some(alpha_blend), write_mask: wgpu::ColorWrites::ALL })],
            }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, ..Default::default() },
            depth_stencil: None, multisample: Default::default(), multiview: None, cache: None,
        });

        let text_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("text"), layout: None,
            vertex:   wgpu::VertexState { module: &text_mod, entry_point: Some("vs"), buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: &text_mod, entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format: surface_fmt, blend: Some(alpha_blend), write_mask: wgpu::ColorWrites::ALL })],
            }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, ..Default::default() },
            depth_stencil: None, multisample: Default::default(), multiview: None, cache: None,
        });

        // Bind groups.
        let spline_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("spline-bg"),
            layout: &spline_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: spline_params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: x_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: y_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: spline_buf.as_entire_binding() },
            ],
        });
        let grid_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("grid-bg"),
            layout: &grid_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: view_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: style_buf.as_entire_binding() },
            ],
        });
        let line_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("line-bg"),
            layout: &line_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: view_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: spline_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: meta_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: axes_buf.as_entire_binding() },
            ],
        });
        let point_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("point-bg"),
            layout: &point_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: view_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: x_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: y_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: meta_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: axes_buf.as_entire_binding() },
            ],
        });

        // Atlas texture.
        let atlas_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("font-atlas"),
            size: wgpu::Extent3d { width: atlas_width, height: atlas_height, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &atlas_texture, mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &atlas_rgba,
            wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(atlas_width * 4), rows_per_image: Some(atlas_height) },
            wgpu::Extent3d { width: atlas_width, height: atlas_height, depth_or_array_layers: 1 },
        );
        let atlas_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("font-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let atlas_tex_view = atlas_texture.create_view(&Default::default());
        let text_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("text-bg"),
            layout: &text_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: text_inst_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&atlas_tex_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&atlas_sampler) },
            ],
        });

        // Parse char metrics from JS (3 floats per char: uMin, uMax, widthPx).
        let char_count = atlas_meta.len() / 3;
        let char_meta: Vec<CharMeta> = (0..char_count).map(|i| CharMeta {
            u_min: atlas_meta[i * 3],
            u_max: atlas_meta[i * 3 + 1],
            width: atlas_meta[i * 3 + 2],
        }).collect();

        let css_w = canvas.client_width().max(1) as f32;
        let css_h = canvas.client_height().max(1) as f32;

        let inner = ChartInner {
            surface, device, queue, surface_fmt, alpha_mode,
            surface_w: canvas_w, surface_h: canvas_h,
            vdv2, series_count, point_count: pc, axis_count,
            view_buf, style_buf, x_buf, y_buf, meta_buf, axes_buf,
            spline_params_buf, spline_buf,
            text_inst_buf, atlas_texture, atlas_sampler, char_meta, atlas_h: atlas_h_px,
            spline_pipeline, grid_pipeline, line_pipeline, point_pipeline, text_pipeline,
            spline_bind, grid_bind, line_bind, point_bind, text_bind,
            css_width: css_w, css_height: css_h, pixel_ratio,
            data_min_x: 0.0, data_max_x: 1.0,
            axis_min_y: vec![0.0; axis_count],
            axis_max_y: vec![1.0; axis_count],
            view_staging: [0u8; VIEW_BYTES as usize],
            axes_staging: vec![0.0; axis_count * 8],
            line_series: vec![], point_series: vec![],
            view_initialized: false, text_char_count: 0, render_queued: false,
            background, grid_color,
        };

        Ok(Rc::new(RefCell::new(inner)))
    }

    // ---- data upload ---------------------------------------------------------

    pub fn upload_data(&mut self) {
        self.queue.write_buffer(&self.x_buf,    0, format::x_bytes(&self.vdv2));
        self.queue.write_buffer(&self.y_buf,    0, format::y_bytes(&self.vdv2));
        self.queue.write_buffer(&self.meta_buf, 0, format::meta_bytes(&self.vdv2));
        self.line_series  = (0..self.series_count).filter(|&s| format::meta_slice(&self.vdv2)[s * 12 + 4] > 0.0).collect();
        self.point_series = (0..self.series_count).filter(|&s| format::meta_slice(&self.vdv2)[s * 12 + 7] > 0.0).collect();
    }

    pub fn upload_axes(&mut self) {
        for a in 0..self.axis_count {
            let y_min = self.axis_min_y[a];
            let y_max = self.axis_max_y[a];
            let range = (y_max - y_min).max(1e-30);
            let o = a * 8;
            self.axes_staging[o]     = 2.0 / range;
            self.axes_staging[o + 1] = (y_min + y_max) * 0.5;
            self.axes_staging[o + 2] = 0.0;
            self.axes_staging[o + 3] = 0.0;
            let c = format::get_axis_color(&self.vdv2, a);
            self.axes_staging[o + 4] = c[0];
            self.axes_staging[o + 5] = c[1];
            self.axes_staging[o + 6] = c[2];
            self.axes_staging[o + 7] = c[3];
        }
        self.queue.write_buffer(&self.axes_buf, 0, bytemuck::cast_slice(&self.axes_staging));
    }

    // ---- spline bake ---------------------------------------------------------

    pub fn bake_spline(&mut self) {
        let total_samples = SUBDIVS as usize * (self.point_count - 1) + 1;
        let params = [self.point_count as u32, self.series_count as u32, total_samples as u32, 0u32];
        self.queue.write_buffer(&self.spline_params_buf, 0, bytemuck::cast_slice(&params));

        let total_work = self.series_count * total_samples;
        let groups = ((total_work + 63) / 64) as u32;
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let cp_desc = wgpu::ComputePassDescriptor { label: Some("spline"), timestamp_writes: None };
            let mut cp = enc.begin_compute_pass(&cp_desc);
            cp.set_pipeline(&self.spline_pipeline);
            cp.set_bind_group(0, &self.spline_bind, &[]);
            cp.dispatch_workgroups(groups, 1, 1);
        }
        self.queue.submit([enc.finish()]);
    }

    // ---- view ----------------------------------------------------------------

    pub fn reset_view(&mut self) {
        let (x_min, x_max, _, _) = format::get_extents(&self.vdv2);
        let pad_x = (x_max - x_min) * 0.02;
        let pad_x = if pad_x == 0.0 { 1.0 } else { pad_x };
        self.data_min_x = x_min - pad_x;
        self.data_max_x = x_max + pad_x;

        let ax = format::get_axis_cpu_extents(&self.vdv2);
        for a in 0..self.axis_count {
            let (y_min, y_max) = ax[a];
            let pad_y = ((y_max - y_min) * 0.05).max(1e-30);
            self.axis_min_y[a] = y_min - pad_y;
            self.axis_max_y[a] = y_max + pad_y;
        }

        let dpr = self.pixel_ratio;
        let w = (self.css_width  * dpr).max(1.0) as u32;
        let h = (self.css_height * dpr).max(1.0) as u32;
        self.upload_axes();
        self.write_view_all(w, h);
        self.view_initialized = true;
        self.update_labels();
    }

    pub fn resize(&mut self, css_w: f32, css_h: f32, dpr: f32) {
        self.css_width   = css_w;
        self.css_height  = css_h;
        self.pixel_ratio = dpr;
        if self.view_initialized {
            let w = (css_w * dpr).max(1.0) as u32;
            let h = (css_h * dpr).max(1.0) as u32;
            self.write_view_all(w, h);
        }
        self.update_labels();
    }

    pub fn pan(&mut self, dx_css: f32) {
        let gutter_l = self.axis_count as f32 * PX_PER_AXIS;
        let plot_w = (self.css_width - gutter_l).max(1.0);
        let dx = (dx_css / plot_w) * (self.data_max_x - self.data_min_x);
        self.data_min_x -= dx;
        self.data_max_x -= dx;
        self.write_view_offset();
        self.update_labels();
    }

    pub fn zoom_at(&mut self, css_x: f32, factor: f32) {
        let gutter_l  = self.axis_count as f32 * PX_PER_AXIS;
        let plot_w    = (self.css_width - gutter_l).max(1.0);
        let fx        = ((css_x - gutter_l) / plot_w).max(0.0);
        let ax        = self.data_min_x + fx * (self.data_max_x - self.data_min_x);
        self.data_min_x = ax + (self.data_min_x - ax) / factor;
        self.data_max_x = ax + (self.data_max_x - ax) / factor;
        self.write_view_scale_offset_step();
        self.update_labels();
    }

    pub fn zoom_to_range(&mut self, x0: f32, x1: f32) {
        if (x1 - x0).abs() < 1e-10 { return; }
        self.data_min_x = x0.min(x1);
        self.data_max_x = x0.max(x1);
        self.write_view_scale_offset_step();
        self.update_labels();
    }

    pub fn pick_x_point(&self, css_x: f32) -> i32 {
        let gutter_l = self.axis_count as f32 * PX_PER_AXIS;
        let plot_w   = (self.css_width - gutter_l).max(1.0);
        let data_x   = self.data_min_x + ((css_x - gutter_l).max(0.0) / plot_w) * (self.data_max_x - self.data_min_x);
        let x = format::x_slice(&self.vdv2);
        pick_x(x, data_x) as i32
    }

    // ---- view uniform helpers ------------------------------------------------

    fn gutter_left(&self) -> f32 { self.axis_count as f32 * PX_PER_AXIS }

    fn view_scale_offset(&self) -> (f32, f32) {
        compute_x_scale_offset(
            self.data_min_x, self.data_max_x,
            self.css_width,
            self.gutter_left(), 0.0,
        )
    }

    fn current_step_x(&self) -> f32 {
        nice_step(self.data_max_x - self.data_min_x, self.css_width, 100.0)
    }

    fn write_view_offset(&mut self) {
        let (_, offset_x) = self.view_scale_offset();
        let vf: &mut [f32] = bytemuck::cast_slice_mut(&mut self.view_staging);
        vf[2] = offset_x;
        self.queue.write_buffer(&self.view_buf, 8, &self.view_staging[8..16]);
    }

    fn write_view_scale_offset_step(&mut self) {
        let (scale_x, offset_x) = self.view_scale_offset();
        let step_x = self.current_step_x();
        let vf: &mut [f32] = bytemuck::cast_slice_mut(&mut self.view_staging);
        vf[0] = scale_x; vf[1] = 0.0;
        vf[2] = offset_x; vf[3] = 0.0;
        vf[6] = step_x; vf[7] = 0.0;
        self.queue.write_buffer(&self.view_buf, 0, &self.view_staging[..32]);
    }

    fn write_view_all(&mut self, w: u32, h: u32) {
        let (scale_x, offset_x) = self.view_scale_offset();
        let step_x = self.current_step_x();
        {
            let vf: &mut [f32] = bytemuck::cast_slice_mut(&mut self.view_staging);
            vf[0] = scale_x; vf[1] = 0.0;
            vf[2] = offset_x; vf[3] = 0.0;
            vf[4] = w as f32; vf[5] = h as f32;
            vf[6] = step_x; vf[7] = 0.0;
        }
        {
            let vu: &mut [u32] = bytemuck::cast_slice_mut(&mut self.view_staging);
            vu[8] = self.point_count as u32;
            vu[9] = self.series_count as u32;
        }
        self.queue.write_buffer(&self.view_buf, 0, &self.view_staging);
        self.maybe_reconfigure_surface(w, h);
    }

    fn maybe_reconfigure_surface(&mut self, w: u32, h: u32) {
        if w == self.surface_w && h == self.surface_h { return; }
        self.surface_w = w;
        self.surface_h = h;
        self.surface.configure(&self.device, &wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: self.surface_fmt,
            width: w, height: h,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: self.alpha_mode,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        });
    }

    // ---- autoscale (CPU, visible X window) ----------------------------------

    pub fn update_axis_extents_cpu(&mut self) {
        let extents = format::compute_visible_axis_extents(
            &self.vdv2, self.data_min_x, self.data_max_x,
        );
        let mut any_changed = false;
        for a in 0..self.axis_count {
            let (y_min, y_max) = extents[a];
            if !y_min.is_finite() || !y_max.is_finite() || y_max <= y_min { continue; }
            let pad = ((y_max - y_min) * 0.05).max(0.5);
            self.axis_min_y[a] = y_min - pad;
            self.axis_max_y[a] = y_max + pad;
            any_changed = true;
        }
        if any_changed {
            self.upload_axes();
            self.update_labels();
        }
    }

    // ---- labels --------------------------------------------------------------

    pub fn update_labels(&mut self) {
        if !self.view_initialized || self.char_meta.is_empty() { return; }
        let step_x = {
            let vf: &[f32] = bytemuck::cast_slice(&self.view_staging);
            vf[6]
        };
        let (data, count) = build_labels(
            &self.vdv2, &self.char_meta, self.atlas_h,
            self.css_width, self.css_height,
            self.data_min_x, self.data_max_x, step_x,
            &self.axis_min_y, &self.axis_max_y,
        );
        self.text_char_count = count as u32;
        if count > 0 {
            let byte_len = count * TEXT_INST_FLOATS * 4;
            self.queue.write_buffer(&self.text_inst_buf, 0, bytemuck::cast_slice(&data[..count * TEXT_INST_FLOATS]));
            let _ = byte_len;
        }
    }

    // ---- render --------------------------------------------------------------

    pub fn render(&mut self) {
        let dpr = self.pixel_ratio;
        let w = (self.css_width  * dpr).max(1.0) as u32;
        let h = (self.css_height * dpr).max(1.0) as u32;
        self.maybe_reconfigure_surface(w, h);

        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(e) => {
                web_sys::console::warn_1(&wasm_bindgen::JsValue::from_str(&format!("get_current_texture: {e:?}")));
                return;
            }
        };
        let view = output.texture.create_view(&Default::default());

        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("render") });
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: self.background[0] as f64,
                            g: self.background[1] as f64,
                            b: self.background[2] as f64,
                            a: self.background[3] as f64,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // Grid.
            pass.set_pipeline(&self.grid_pipeline);
            pass.set_bind_group(0, &self.grid_bind, &[]);
            pass.draw(0..3, 0..1);

            // Lines (spline).
            if !self.line_series.is_empty() && self.point_count >= 2 {
                let x = format::x_slice(&self.vdv2);
                let (first_s, last_s) = visible_samples(x, self.data_min_x, self.data_max_x, SUBDIVS as usize);
                let count = (last_s as i32 - first_s as i32 + 1).max(0) as u32;
                if count > 0 {
                    let v0 = (2 * first_s) as u32;
                    pass.set_pipeline(&self.line_pipeline);
                    pass.set_bind_group(0, &self.line_bind, &[]);
                    for &s in &self.line_series {
                        // vertex range covers [v0, v0 + 2*count); instance selects series
                        pass.draw(v0..(v0 + 2 * count), s as u32..(s as u32 + 1));
                    }
                }
            }

            // Points.
            if !self.point_series.is_empty() {
                let x = format::x_slice(&self.vdv2);
                let (first_pt, last_pt) = visible_points(x, self.data_min_x, self.data_max_x);
                let pt_count = (last_pt as i32 - first_pt as i32 + 1).max(0) as u32;
                if pt_count > 0 {
                    pass.set_pipeline(&self.point_pipeline);
                    pass.set_bind_group(0, &self.point_bind, &[]);
                    for &s in &self.point_series {
                        let base = (s * self.point_count + first_pt) as u32;
                        // 6 vertices per point-quad; instance_index encodes series*N + pointIdx
                        pass.draw(0..6, base..(base + pt_count));
                    }
                }
            }

            // Text labels.
            if self.text_char_count > 0 {
                pass.set_pipeline(&self.text_pipeline);
                pass.set_bind_group(0, &self.text_bind, &[]);
                pass.draw(0..6, 0..self.text_char_count);
            }
        }

        self.queue.submit([enc.finish()]);
        output.present();
    }

    pub fn request_render(inner_rc: &Rc<RefCell<ChartInner>>) {
        {
            let inner = inner_rc.borrow();
            if inner.render_queued { return; }
        }
        inner_rc.borrow_mut().render_queued = true;

        let inner_clone = Rc::clone(inner_rc);
        let closure = wasm_bindgen::closure::Closure::once(move || {
            let mut inner = inner_clone.borrow_mut();
            inner.render_queued = false;
            inner.render();
        });
        web_sys::window()
            .expect("no window")
            .request_animation_frame(closure.as_ref().unchecked_ref())
            .unwrap();
        closure.forget();
    }

}
