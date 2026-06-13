/* tslint:disable */
/* eslint-disable */

/**
 * Opaque chart handle exposed to JavaScript.
 */
export class Chart {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    buffer_len(): number;
    /**
     * Pointer into WASM linear memory where the VDV2 buffer lives.
     * JS uses this + memory.buffer to build zero-copy typed-array views.
     */
    buffer_ptr(): number;
    /**
     * Async factory. `groups` is a Uint32Array of per-axis series counts.
     * `atlas_meta` is a Float32Array of [uMin, uMax, widthPx] triplets for each
     * character in TEXT_CHARS order (built once in JS from Canvas-2D).
     * Returns a Promise<Chart>.
     */
    static create(canvas: HTMLCanvasElement, groups: Uint32Array, point_count: number, atlas_rgba: Uint8Array, atlas_width: number, atlas_height: number, atlas_meta: Float32Array, atlas_h_px: number, bg_r: number, bg_g: number, bg_b: number, bg_a: number, grid_r: number, grid_g: number, grid_b: number, grid_a: number, pixel_ratio: number): Promise<Chart>;
    pan(dx_css: number): void;
    /**
     * Return the nearest data-point index to a CSS-pixel X coordinate.
     */
    pick_x(css_x: number): number;
    /**
     * Explicit render call (e.g. from a JS requestAnimationFrame loop).
     */
    render(): void;
    request_render(): void;
    reset_view(): void;
    resize(css_w: number, css_h: number, dpr: number): void;
    /**
     * Full reset: upload data, recompute extents, reset view, render.
     */
    set_data(): void;
    /**
     * Upload the VDV2 buffer (x, y, meta) to GPU buffers and re-bake the spline.
     */
    upload_data(): void;
    zoom_at(css_x: number, factor: number): void;
    /**
     * Zoom to a CSS-pixel X range (right-drag selection from JS shim).
     */
    zoom_to_css_range(css_x0: number, css_x1: number): void;
}

/**
 * Recompute x/y global extents in-place in a VDV2 buffer at the given ptr/len.
 * Called from JS after filling x and y data.
 */
export function vdv2_recompute_extents(ptr: number, len: number): void;

/**
 * Return the WebAssembly.Memory object so JS can build zero-copy typed-array
 * views over the WASM linear memory at ptr/len offsets from buffer_ptr().
 */
export function wasm_memory(): any;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_chart_free: (a: number, b: number) => void;
    readonly chart_buffer_len: (a: number) => number;
    readonly chart_buffer_ptr: (a: number) => number;
    readonly chart_create: (a: any, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number, r: number, s: number, t: number) => any;
    readonly chart_pan: (a: number, b: number) => void;
    readonly chart_pick_x: (a: number, b: number) => number;
    readonly chart_render: (a: number) => void;
    readonly chart_request_render: (a: number) => void;
    readonly chart_reset_view: (a: number) => void;
    readonly chart_resize: (a: number, b: number, c: number, d: number) => void;
    readonly chart_set_data: (a: number) => void;
    readonly chart_upload_data: (a: number) => void;
    readonly chart_zoom_at: (a: number, b: number, c: number) => void;
    readonly chart_zoom_to_css_range: (a: number, b: number, c: number) => void;
    readonly wasm_memory: () => any;
    readonly vdv2_recompute_extents: (a: number, b: number) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h57bcbce08f24a31e: (a: number, b: number, c: any) => [number, number];
    readonly wasm_bindgen__convert__closures_____invoke__h03788091b22bd43a: (a: number, b: number, c: any, d: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h1b022b9ab2fd2a7b: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h1b022b9ab2fd2a7b_2: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h9d8ef805b3b9b4bd: (a: number, b: number) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_destroy_closure: (a: number, b: number) => void;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
