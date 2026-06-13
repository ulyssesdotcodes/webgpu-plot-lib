import { LineChart, type HoverInfo } from './chart.js';
import { generateLines } from './data.js';

function fail(msg: string): never {
  document.body.innerHTML = `<pre style="color:red;padding:1em">${msg}</pre>`;
  throw new Error(msg);
}

// Determine engine from query string: ?engine=rust
const params  = new URLSearchParams(location.search);
const useRust = params.get('engine') === 'rust';

async function main(): Promise<void> {
  const canvas = document.querySelector('canvas');
  if (!canvas) fail('no canvas element');

  const tooltip = document.getElementById('tooltip');
  if (!tooltip) fail('no #tooltip element');

  const GROUPS       = [3, 2, 2] as const;
  const POINT_COUNT  = 4096 * 8;

  if (useRust) {
    await runRust(canvas, tooltip, [...GROUPS], POINT_COUNT);
  } else {
    await runTs(canvas, tooltip, [...GROUPS], POINT_COUNT);
  }
}

// ---- TypeScript engine (original) ------------------------------------------

async function runTs(
  canvas: HTMLCanvasElement,
  tooltip: HTMLElement,
  groups: number[],
  pointCount: number,
): Promise<void> {
  const data = generateLines(groups, pointCount);

  const chart = await LineChart.create(canvas, data, {
    onHover: (info: HoverInfo | null) => {
      if (!info) { tooltip.style.display = 'none'; return; }
      const lines = [`<b>x = ${info.x.toFixed(4)}</b>`];
      for (let s = 0; s < info.ys.length; s++) {
        const r = Math.round(data.meta[s * 12 + 0]! * 255);
        const g = Math.round(data.meta[s * 12 + 1]! * 255);
        const b = Math.round(data.meta[s * 12 + 2]! * 255);
        lines.push(`<span style="color:rgb(${r},${g},${b})">■</span> ${info.ys[s]!.toFixed(4)}`);
      }
      tooltip.style.display = 'block';
      tooltip.innerHTML = lines.join('<br>');
    },
  });

  canvas.addEventListener('mousemove', (e) => {
    tooltip.style.left = `${e.clientX + 12}px`;
    tooltip.style.top  = `${e.clientY + 12}px`;
  });

  (window as unknown as { chart: typeof chart; data: typeof data }).chart = chart;
  (window as unknown as { chart: typeof chart; data: typeof data }).data  = data;
}

// ---- Rust/wgpu engine -------------------------------------------------------

// TEXT_CHARS order must match labels.rs — kept in sync here.
const TEXT_CHARS = '0123456789.-█';

/** Build the font atlas the same way chart.ts does, returning RGBA bytes + metrics. */
function buildAtlas(): {
  rgba: Uint8Array;
  width: number;
  height: number;
  meta: Float32Array;   // [uMin, uMax, widthPx] × charCount
  atlasH: number;
} {
  const font = '12px monospace';
  const pad  = 1;

  const mc = document.createElement('canvas');
  mc.width = 1; mc.height = 1;
  const mctx = mc.getContext('2d')!;
  mctx.font = font;

  const metrics = mctx.measureText('0');
  const atlasH  = Math.ceil(metrics.actualBoundingBoxAscent + metrics.actualBoundingBoxDescent) + 4;

  const chars      = Array.from(TEXT_CHARS);
  const slotWidths = chars.map(c => Math.ceil(mctx.measureText(c).width) + pad * 2);
  const totalW     = slotWidths.reduce((a, b) => a + b, 0);
  const texW       = Math.ceil(totalW / 64) * 64;

  const ac = document.createElement('canvas');
  ac.width = texW; ac.height = atlasH;
  const ctx = ac.getContext('2d')!;
  ctx.font         = font;
  ctx.fillStyle    = 'white';
  ctx.textBaseline = 'middle';

  let x   = 0;
  const meta = new Float32Array(chars.length * 3);
  for (let i = 0; i < chars.length; i++) {
    const w = slotWidths[i]!;
    ctx.fillText(chars[i]!, x + pad, atlasH / 2);
    meta[i * 3]     = x / texW;
    meta[i * 3 + 1] = (x + w) / texW;
    meta[i * 3 + 2] = w;
    x += w;
  }

  const imgData = ctx.getImageData(0, 0, texW, atlasH);
  return { rgba: new Uint8Array(imgData.data.buffer), width: texW, height: atlasH, meta, atlasH };
}

/** Fill the VDV2 buffer (allocated in WASM memory) with test data matching generateLines. */
function fillWasmVdv2(
  memory: WebAssembly.Memory,
  ptr: number,
  len: number,
  groups: number[],
  pointCount: number,
): void {
  const buf    = memory.buffer;
  const hdrU   = new Uint32Array(buf, ptr, 16);
  const hdrF   = new Float32Array(buf, ptr, 16);

  const xOff   = hdrU[5]!;
  const yOff   = hdrU[6]!;
  const metaOff = hdrU[7]!;
  const axesOff = hdrU[13]!;
  const seriesCount = hdrU[3]!;

  const xView    = new Float32Array(buf, ptr + xOff,    pointCount);
  const yView    = new Float32Array(buf, ptr + yOff,    seriesCount * pointCount);
  const metaView = new Float32Array(buf, ptr + metaOff, seriesCount * 12);
  const axesView = new Float32Array(buf, ptr + axesOff, groups.length * 8);

  // X axis: 0..1
  for (let i = 0; i < pointCount; i++) xView[i] = i / (pointCount - 1);

  // Y data, styles, axis colors (mirrors data.ts generateLines)
  let s = 0;
  for (let a = 0; a < groups.length; a++) {
    const groupAmp = Math.pow(10, a * 2);
    for (let g = 0; g < groups[a]!; g++, s++) {
      const phase = (s / seriesCount) * Math.PI * 2;
      const freq  = 1 + s * 0.7;
      const amp   = groupAmp * (0.7 + 0.2);   // deterministic amp
      const drift = (g - groups[a]! / 2) * groupAmp * 0.4;
      let walk = 0;
      // Simple deterministic walk using multiplicative LCG (no BigInt).
      let seed = (s + 1) * 1664525 + 1013904223;
      for (let i = 0; i < pointCount; i++) {
        const t = xView[i]!;
        seed = (seed * 1664525 + 1013904223) >>> 0;
        walk += (seed / 0x100000000 - 0.5) * 0.05 * groupAmp;
        yView[s * pointCount + i] = drift + amp * Math.sin(phase + freq * t * Math.PI * 2) + walk;
      }

      const hue = s / seriesCount;
      const [r, gr, b] = hslToRgb(hue, 0.7, 0.6);
      const hasPoints = g === 0;
      const o = s * 12;
      metaView[o]     = r;  metaView[o + 1] = gr; metaView[o + 2] = b; metaView[o + 3] = 1;
      metaView[o + 4] = hasPoints ? 0 : 2.5;
      metaView[o + 5] = a;
      const SHAPES = [1, 2, 3]; // circle, triangle, square
      metaView[o + 6] = hasPoints ? 5 : 0;
      metaView[o + 7] = hasPoints ? SHAPES[a % 3]! : 0;
      metaView[o + 8] = r; metaView[o + 9] = gr; metaView[o + 10] = b; metaView[o + 11] = 1;

      if (g === 0) {
        const ao = a * 8;
        axesView[ao] = r; axesView[ao + 1] = gr; axesView[ao + 2] = b; axesView[ao + 3] = 1;
      }
    }
  }

  // Global extents (x is 0..1; y: approximate)
  hdrF[8]  = 0; hdrF[9]  = 1;
  let gYMin = Infinity, gYMax = -Infinity;
  for (let i = 0; i < yView.length; i++) {
    if (yView[i]! < gYMin) gYMin = yView[i]!;
    if (yView[i]! > gYMax) gYMax = yView[i]!;
  }
  hdrF[10] = gYMin; hdrF[11] = gYMax;
}

async function runRust(
  canvas: HTMLCanvasElement,
  tooltip: HTMLElement,
  groups: number[],
  pointCount: number,
): Promise<void> {
  // Dynamic import so the wasm module is only loaded in rust mode.
  const wasmMod = await import('../public/pkg/webgpu_plot_lib.js') as {
    default: () => Promise<void>;
    Chart: {
      create(
        canvas: HTMLCanvasElement,
        groups: Uint32Array,
        point_count: number,
        atlas_rgba: Uint8Array,
        atlas_width: number,
        atlas_height: number,
        atlas_meta: Float32Array,
        atlas_h_px: number,
        bg_r: number, bg_g: number, bg_b: number, bg_a: number,
        grid_r: number, grid_g: number, grid_b: number, grid_a: number,
        pixel_ratio: number,
      ): Promise<RustChart>;
    };
    wasm_memory(): WebAssembly.Memory;
  };

  await wasmMod.default(); // init wasm
  console.log('[rust] wasm module initialized');

  const atlas = buildAtlas();

  const chart = await wasmMod.Chart.create(
    canvas,
    new Uint32Array(groups),
    pointCount,
    atlas.rgba,
    atlas.width,
    atlas.height,
    atlas.meta,
    atlas.atlasH,
    0.07, 0.08, 0.11, 1,   // background
    0.22, 0.24, 0.28, 1,   // grid color
    1,                      // pixelRatio
  );

  // Write test data into the WASM-memory VDV2 buffer (zero-copy).
  const memory = wasmMod.wasm_memory();
  const ptr = chart.buffer_ptr();
  const len = chart.buffer_len();
  fillWasmVdv2(memory, ptr, len, groups, pointCount);

  // Upload to GPU + initial render.
  chart.reset_view();

  // Sync CSS size to physical canvas.
  const syncSize = () => {
    const w = canvas.clientWidth  || 1;
    const h = canvas.clientHeight || 1;
    chart.resize(w, h, 1);
  };
  syncSize();
  new ResizeObserver(syncSize).observe(canvas);

  // Event wiring (thin shim — keeps the WASM boundary computational).
  let dragging = false;
  let lastX    = 0;
  let rightDragging    = false;
  let rightStartX      = 0;
  let selectionEl: HTMLDivElement | null = null;

  canvas.addEventListener('contextmenu', e => e.preventDefault());

  canvas.addEventListener('mousedown', (e) => {
    if (e.button === 0) { dragging = true; lastX = e.clientX; canvas.style.cursor = 'grabbing'; }
    else if (e.button === 2) {
      e.preventDefault();
      rightDragging = true; rightStartX = e.clientX;
      selectionEl = document.createElement('div');
      selectionEl.style.cssText = 'position:fixed;pointer-events:none;background:rgba(100,160,255,0.15);border:1px solid rgba(100,160,255,0.7);box-sizing:border-box;z-index:9999';
      document.body.appendChild(selectionEl);
    }
  });

  window.addEventListener('mouseup', (e) => {
    if (dragging) { dragging = false; canvas.style.cursor = ''; }
    if (rightDragging) {
      rightDragging = false;
      if (selectionEl) { selectionEl.remove(); selectionEl = null; }
      canvas.style.cursor = '';
      const dx = e.clientX - rightStartX;
      if (Math.abs(dx) > 4) {
        const rect2 = canvas.getBoundingClientRect();
        chart.zoom_to_css_range(
          Math.min(rightStartX, e.clientX) - rect2.left,
          Math.max(rightStartX, e.clientX) - rect2.left,
        );
      }
    }
  });

  window.addEventListener('mousemove', (e) => {
    if (dragging) {
      chart.pan(e.clientX - lastX);  // pan is snake_case in wgpu bindings
      lastX = e.clientX;
    } else if (rightDragging && selectionEl) {
      const rect = canvas.getBoundingClientRect();
      const PX_PER_AXIS = 55;
      const gutter = groups.length * PX_PER_AXIS;
      const plotL = rect.left + gutter;
      const plotR = rect.right;
      const x0 = Math.max(plotL, Math.min(plotR, Math.min(rightStartX, e.clientX)));
      const x1 = Math.max(plotL, Math.min(plotR, Math.max(rightStartX, e.clientX)));
      Object.assign(selectionEl.style, {
        left: `${x0}px`, top: `${rect.top}px`,
        width: `${x1 - x0}px`, height: `${rect.height}px`,
        display: x1 - x0 > 0 ? 'block' : 'none',
      });
    }
  });

  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const rect   = canvas.getBoundingClientRect();
    const factor = Math.exp(-e.deltaY * 0.0015);
    chart.zoom_at(e.clientX - rect.left, factor);
  }, { passive: false });

  canvas.addEventListener('dblclick', () => chart.reset_view());

  canvas.addEventListener('mousemove', (e) => {
    tooltip.style.left = `${e.clientX + 12}px`;
    tooltip.style.top  = `${e.clientY + 12}px`;
  });

  console.log('[rust] engine running');
  (window as unknown as { rustChart: typeof chart }).rustChart = chart;
}

// Type shim for the wasm Chart handle (dynamic import, no generated TS types needed at build time).
interface RustChart {
  buffer_ptr(): number;
  buffer_len(): number;
  reset_view(): void;
  resize(css_w: number, css_h: number, dpr: number): void;
  pan(dx_css: number): void;
  zoom_at(css_x: number, factor: number): void;
  zoom_to_css_range(css_x0: number, css_x1: number): void;
  pick_x(css_x: number): number;
  render(): void;
}

function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  const a = s * Math.min(l, 1 - l);
  const f = (n: number) => {
    const k = (n + h * 12) % 12;
    return l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1));
  };
  return [f(0), f(8), f(4)];
}

main();
