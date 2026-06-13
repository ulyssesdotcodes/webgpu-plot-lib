# Rust/wgpu bindings for the line-chart drawing library — Implementation Plan

> Implementation plan for porting the TypeScript WebGPU line-chart drawing
> library to Rust using the [`wgpu`](https://github.com/gfx-rs/wgpu) crate
> (compiled to WASM via `wasm-bindgen`). Hand this to an implementation agent.

## Context

`webgpu-plot-lib` is a TypeScript WebGPU line chart (`src/chart.ts` `LineChart`)
built on a self-describing single-`ArrayBuffer` wire format ("VDV2",
`src/format.ts`). Data lives in one transferable buffer: header + shared X column
+ N×M Y matrix + per-series meta + per-axis config. Rendering uses WGSL shaders
in `public/shaders/` (grid, lines, points, text) plus two compute shaders
(`spline.wgsl` bakes Catmull-Rom splines; `extent.wgsl` reduces per-axis visible
Y min/max for autoscale).

We want a Rust implementation of this drawing library using the `wgpu` crate,
compiled to WASM and exposed to JS via `wasm-bindgen`. The explicit priority is
**minimizing transfer cost**: operate on `ArrayBuffer`/`SharedArrayBuffer` and
typed-array views over WASM linear memory rather than marshalling big JS objects
across the boundary.

Decisions confirmed with the user:
- **Coexist as a new `rust/` crate alongside the existing TS** (TS stays as the
  reference implementation / fallback; selectable at runtime).
- **Maximize WGSL/GPU compute.** Reuse the existing `.wgsl` verbatim (wgpu reads
  WGSL natively) and additionally move the last heavy CPU pass — initial
  full-data extents — onto the GPU.
- **Threading:** main-thread + wasm-memory baseline first (Phase 1), then
  Worker + OffscreenCanvas + SharedArrayBuffer for best large-data performance
  (Phase 2). The Rust API is context-agnostic so the same code runs in both.

### Background: how much can move into WGSL?

Almost all per-data-point math already lives on the GPU: Catmull-Rom spline
baking (`spline.wgsl`), per-axis visible-Y autoscale (`extent.wgsl`), and
thick-line/point/grid/text rendering. The only remaining O(N·M) CPU pass is the
*initial* full-data extent computation (`recomputeExtents`/`getAxisCpuExtents` in
`src/format.ts`) — moved to the GPU here by reusing `extent.wgsl` over an
infinite X window. What's left on the CPU is all O(small): view scale/offset,
`niceStep`/tick-label formatting, hover pick (O(log N) binary search), and
font-atlas rasterization. Those are not worth moving to WGSL.

### Background: which threading model is most performant for large data?

A **Web Worker + OffscreenCanvas + SharedArrayBuffer** is most performant for
large/streaming datasets: the whole render loop and data ingestion run off the
main thread (no UI jank during buffer build/upload), and a SharedArrayBuffer lets
a producer write into the VDV2 buffer with zero copy. The cost is COOP/COEP
cross-origin-isolation headers (the current `http-server` can't set them, so we
add a tiny header-setting dev server) plus more wiring. Main-thread + wasm-memory
is simpler and still avoids object marshalling but can stall on large updates.
Hence: ship main-thread first, then the Worker/SAB path.

## Design principles (transfer cost)

- The VDV2 buffer is allocated **inside WASM linear memory** (Rust `Vec<u8>`).
  Rust exposes its `ptr`+`len`; JS builds `Float32Array`/`Uint32Array` views over
  `wasm.memory.buffer` at those offsets and writes data **in place** — zero copy,
  no big JS objects. Rust uploads from that same memory straight to GPU buffers.
- No structured-object API. Style/axis edits are numeric setters that write the
  `meta`/`axes` regions directly (mirroring `setStyle`/`setAxisConfig`). Hover
  results are written into a small preallocated scratch `Float32Array` view, not
  returned as objects.
- Phase 2: the VDV2 buffer is backed by a `SharedArrayBuffer` shared between the
  render Worker and any producer thread; same byte layout, so no code changes in
  the parse/build core.

## Crate layout (`rust/`)

- `Cargo.toml` — `crate-type = ["cdylib", "rlib"]`; deps: `wgpu`, `wasm-bindgen`,
  `wasm-bindgen-futures` (async `requestAdapter`/`requestDevice`), `js-sys`,
  `web-sys` (HtmlCanvasElement, OffscreenCanvas), `bytemuck`,
  `console_error_panic_hook`. `format.rs` builds on native too for `cargo test`.
- `src/format.rs` — VDV2 over `&[u8]`/`&mut [u8]`: `MAGIC`/`VERSION`/strides,
  `create_series_buffer(groups, point_count) -> Vec<u8>`, `view` accessors,
  `set_style`, `set_axis_config`, `get_series_axis`, header extent get/set. Direct
  port of `src/format.ts`. Platform-agnostic; covered by `#[cfg(test)]`.
- `src/chart.rs` — port of `LineChart`: wgpu device/queue, all GPU buffers and
  pipelines (1:1 with `chart.ts` `buildResources`), bind groups, `bake_spline`,
  `request_axis_extents` + readback, `render`. Shaders via `include_str!` of the
  existing `public/shaders/*.wgsl` (single source of truth — do not fork WGSL;
  reference the files in `public/shaders/`).
- `src/view.rs` — view math: `compute_x_scale_offset`, `pan`, `zoom_at`,
  `nice_step`, `format_tick`, `visible_points`/`visible_samples`, `pick_x`.
- `src/labels.rs` — port of `updateLabels` text-instance builder.
- `src/lib.rs` — `#[wasm_bindgen]` surface (see below).

## JS-facing API (`#[wasm_bindgen]`)

Thin, numeric, buffer-oriented:
- `Chart::create(canvas, vdv2_ptr_or_view, atlas_rgba, atlas_meta, opts) -> Promise`
- `buffer_ptr()`, `buffer_len()` so JS makes zero-copy views into WASM memory.
- `reset_view()`, `resize(css_w, css_h, dpr)`, `pan(dx_css)`,
  `zoom_at(css_x, factor)`, `zoom_to_range(x0, y0)`, `request_render()`.
- `pick_x(css_x) -> i32` (point index; ys written into a shared scratch view).
- `set_style_*` numeric setters writing the meta region; `set_data(ptr,len)`.
- DOM event wiring (mouse/resize/wheel) stays in a small TS shim that calls these
  methods — keeps the WASM boundary computational, not event-driven.

Font atlas: built once in JS (reuse existing `buildAtlas` Canvas-2D logic),
passing the RGBA `Uint8Array` + char-metric floats into `Chart::create`. Keeps
Rust free of a 2D-canvas text dependency and works in a Worker (OffscreenCanvas).

## GPU compute maximization

- Reuse `spline.wgsl` and `extent.wgsl` unchanged.
- **Move initial extents to GPU:** at load/`set_data`, run `extent.wgsl` with
  `xMin=-INF`, `xMax=+INF` to get global per-axis Y extents, plus a tiny reduction
  for global X range — replacing the JS `getAxisCpuExtents`/`recomputeExtents`
  loops. This is the only new GPU work; everything else is already WGSL.
- (Optional, deferred) replace the CPU visible-range binary search with a compute
  pass writing `drawIndirect` args. Low payoff; not in scope unless requested.

## Build & integration

- Add `wasm-pack` (or `cargo` + `wasm-bindgen-cli`) to `flake.nix` `buildInputs`;
  build output to `public/pkg/`. Add an npm `build:wasm` script and a
  `shellHook` step alongside the existing `tsc --watch`.
- `src/index.ts` gains an engine toggle (e.g. `?engine=rust`) that loads the
  wasm-pack glue and constructs `Chart` instead of TS `LineChart`, sharing the
  same `generateLines` VDV2 buffer (written into WASM memory). TS path untouched.
- **Phase 2 only:** add a ~30-line Node static dev server that sets
  `Cross-Origin-Opener-Policy: same-origin` + `Cross-Origin-Embedder-Policy:
  require-corp` (replacing `http-server`), move `Chart` into a Worker with
  `OffscreenCanvas` (`canvas.transferControlToOffscreen()`), and back the VDV2
  buffer with a `SharedArrayBuffer`.

## Phases

1. **Core port (main thread):** `rust/` crate, `format.rs` (+tests), full `wgpu`
   renderer reusing the WGSL, GPU initial-extents, wasm-bindgen API, zero-copy
   wasm-memory data path, JS engine toggle, flake/npm build wiring. Visually and
   behaviorally matches the TS chart.
2. **Worker + OffscreenCanvas + SharedArrayBuffer:** header dev server, move
   render loop into a Worker, SAB-backed VDV2 buffer for best large-data perf.

## Verification

- `cargo test` for `format.rs`: round-trip build → view → edit; assert magic,
  offsets, strides, extents match the TS layout byte-for-byte.
- Build wasm, serve `public/`, open `index.html?engine=rust`: confirm the adapter
  console log, then verify grid/lines/points/text render identically to the TS
  engine; exercise pan (drag), zoom (wheel + right-drag range), double-click
  reset, hover tooltip, and resize.
- Large-data check: bump `generateLines` point count, compare the existing
  `[render] avg cpu … fps` logs between `engine=ts` and `engine=rust`.
- Phase 2: confirm `crossOriginIsolated === true`, chart renders from the Worker,
  and updates written into the SharedArrayBuffer appear without a postMessage copy.

## Key reference files

- `src/format.ts` — VDV2 wire format (port to `rust/src/format.rs`).
- `src/chart.ts` — `LineChart` rendering/compute orchestration (port to
  `rust/src/chart.rs`/`view.rs`/`labels.rs`).
- `public/shaders/{grid,lines,points,text,spline,extent}.wgsl` — reuse verbatim.
- `src/index.ts` — demo entry point; add the engine toggle here.
- `flake.nix` — dev shell; add wasm toolchain + build step.
