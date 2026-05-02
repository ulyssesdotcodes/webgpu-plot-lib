import {
  type SeriesBufferView,
  getExtents,
} from './format.js';

// Layout of the View uniform — kept in sync with both lines.wgsl and grid.wgsl.
//
//   offset 0   vec2f scale
//   offset 8   vec2f offset
//   offset 16  vec2f viewport (physical pixels)
//   offset 24  u32   pointCount
//   offset 28  u32   seriesCount
//   total      32 bytes
const VIEW_BYTES = 32;

// Grid uniform: vec2 step, _pad, vec4 color, vec4 bg = 48 bytes.
const GRID_BYTES = 48;

// Hover uniform: vec2 mousePx, _pad = 16 bytes.
const HOVER_BYTES = 16;

export interface HoverInfo {
  series: number;
  point: number;
  x: number;
  y: number;
  /** Distance in physical pixels from the cursor to the picked point. */
  distancePx: number;
}

export interface LineChartOptions {
  /** Background fill — drawn by the grid pass. */
  background?: [number, number, number, number];
  /** Grid line color. */
  gridColor?: [number, number, number, number];
  /** Pixel distance threshold beyond which a hover hit is treated as a miss. */
  hoverThresholdPx?: number;
  /** Called whenever the hover target changes. `null` means no hit. */
  onHover?: (info: HoverInfo | null) => void;
}

export class LineChart {
  private canvas: HTMLCanvasElement;
  private context: GPUCanvasContext;
  private device: GPUDevice;
  private format: GPUTextureFormat;

  private series: SeriesBufferView;
  private opts: Required<Omit<LineChartOptions, 'onHover'>> & Pick<LineChartOptions, 'onHover'>;

  // GPU resources.
  private viewBuf!: GPUBuffer;
  private gridBuf!: GPUBuffer;
  private hoverBuf!: GPUBuffer;
  private xBuf!: GPUBuffer;
  private yBuf!: GPUBuffer;
  private metaBuf!: GPUBuffer;
  private hoverResult!: GPUBuffer;
  private hoverReadback!: GPUBuffer;
  private extentsParamsBuf!: GPUBuffer;
  private extentsResult!: GPUBuffer;
  private extentsReadback!: GPUBuffer;

  private gridPipeline!: GPURenderPipeline;
  private linePipeline!: GPURenderPipeline;
  private hoverPipeline!: GPUComputePipeline;
  private extentsPipeline!: GPUComputePipeline;
  private gridBind!: GPUBindGroup;
  private lineBind!: GPUBindGroup;
  private hoverBind!: GPUBindGroup;
  private extentsBind!: GPUBindGroup;

  // View state — data-domain rectangle currently visible.
  private dataMinX = 0;
  private dataMaxX = 1;
  private dataMinY = 0;
  private dataMaxY = 1;

  // Mouse state.
  private dragging = false;
  private lastMouse: [number, number] = [0, 0];
  private hoverPending = false;
  private lastHover: HoverInfo | null = null;

  private renderQueued = false;

  static async create(
    canvas: HTMLCanvasElement,
    series: SeriesBufferView,
    options: LineChartOptions = {},
  ): Promise<LineChart> {
    const adapter = await navigator.gpu?.requestAdapter();
    const device = await adapter?.requestDevice();
    if (!device) throw new Error('WebGPU not available');

    const context = canvas.getContext('webgpu');
    if (!context) throw new Error('canvas has no webgpu context');

    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format, alphaMode: 'opaque' });

    const [grid, lines, hover, extents] = await Promise.all([
      fetchShader(device, '/shaders/grid.wgsl', 'grid'),
      fetchShader(device, '/shaders/lines.wgsl', 'lines'),
      fetchShader(device, '/shaders/hover.wgsl', 'hover'),
      fetchShader(device, '/shaders/extents.wgsl', 'extents'),
    ]);

    const chart = new LineChart(canvas, context, device, format, series, options);
    chart.buildResources(grid, lines, hover, extents);
    chart.attachEvents();
    await chart.computeExtentsGPU();
    chart.resetView();
    return chart;
  }

  private constructor(
    canvas: HTMLCanvasElement,
    context: GPUCanvasContext,
    device: GPUDevice,
    format: GPUTextureFormat,
    series: SeriesBufferView,
    options: LineChartOptions,
  ) {
    this.canvas = canvas;
    this.context = context;
    this.device = device;
    this.format = format;
    this.series = series;
    this.opts = {
      background: options.background ?? [0.07, 0.08, 0.11, 1],
      gridColor:  options.gridColor  ?? [0.22, 0.24, 0.28, 1],
      hoverThresholdPx: options.hoverThresholdPx ?? 24,
      ...(options.onHover !== undefined ? { onHover: options.onHover } : {}),
    };

  }

  // ---- public API ----------------------------------------------------------

  /** Reset the visible window to fit all data. */
  resetView(): void {
    const e = getExtents(this.series);
    const padX = (e.xMax - e.xMin) * 0.02 || 1;
    const padY = (e.yMax - e.yMin) * 0.05 || 1;
    this.dataMinX = e.xMin - padX;
    this.dataMaxX = e.xMax + padX;
    this.dataMinY = e.yMin - padY;
    this.dataMaxY = e.yMax + padY;
    this.requestRender();
  }

  /** Replace the underlying data, re-upload to the GPU, recompute extents. */
  async setData(series: SeriesBufferView): Promise<void> {
    if (
      series.seriesCount !== this.series.seriesCount ||
      series.pointCount !== this.series.pointCount
    ) {
      throw new Error('setData: series/point shape must match');
    }
    this.series = series;
    this.uploadData();
    await this.computeExtentsGPU();
    this.requestRender();
  }

  /** Schedule a frame; multiple calls per frame coalesce. */
  requestRender(): void {
    if (this.renderQueued) return;
    this.renderQueued = true;
    requestAnimationFrame(() => {
      this.renderQueued = false;
      this.render();
    });
  }

  // ---- setup ---------------------------------------------------------------

  private buildResources(
    gridMod: GPUShaderModule,
    lineMod: GPUShaderModule,
    hoverMod: GPUShaderModule,
    extentsMod: GPUShaderModule,
  ): void {
    const { device, series } = this;

    // Uniform / data buffers ---
    this.viewBuf  = device.createBuffer({ label: 'view',  size: VIEW_BYTES,  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.gridBuf  = device.createBuffer({ label: 'grid',  size: GRID_BYTES,  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.hoverBuf = device.createBuffer({ label: 'hover', size: HOVER_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    this.xBuf    = device.createBuffer({ label: 'x',    size: series.x.byteLength,    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.yBuf    = device.createBuffer({ label: 'y',    size: series.y.byteLength,    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.metaBuf = device.createBuffer({ label: 'meta', size: series.meta.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });

    this.hoverResult   = device.createBuffer({ label: 'hover-result',   size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.hoverReadback = device.createBuffer({ label: 'hover-readback', size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

    // Extents reduction buffers — 4 × u32 of packed-ordered floats.
    this.extentsParamsBuf = device.createBuffer({ label: 'extents-params',   size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.extentsResult    = device.createBuffer({ label: 'extents-result',   size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.extentsReadback  = device.createBuffer({ label: 'extents-readback', size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

    this.uploadData();

    // Pipelines ---
    this.gridPipeline = device.createRenderPipeline({
      label: 'grid',
      layout: 'auto',
      vertex:   { module: gridMod, entryPoint: 'vs' },
      fragment: { module: gridMod, entryPoint: 'fs', targets: [{ format: this.format }] },
      primitive: { topology: 'triangle-list' },
    });

    this.linePipeline = device.createRenderPipeline({
      label: 'lines',
      layout: 'auto',
      vertex:   { module: lineMod, entryPoint: 'vs' },
      fragment: {
        module: lineMod, entryPoint: 'fs',
        targets: [{
          format: this.format,
          blend: {
            color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one',       dstFactor: 'one-minus-src-alpha', operation: 'add' },
          },
        }],
      },
      primitive: { topology: 'triangle-list' },
    });

    this.hoverPipeline = device.createComputePipeline({
      label: 'hover',
      layout: 'auto',
      compute: { module: hoverMod, entryPoint: 'main' },
    });

    this.extentsPipeline = device.createComputePipeline({
      label: 'extents',
      layout: 'auto',
      compute: { module: extentsMod, entryPoint: 'main' },
    });

    this.gridBind = device.createBindGroup({
      layout: this.gridPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.viewBuf } },
        { binding: 1, resource: { buffer: this.gridBuf } },
      ],
    });

    this.lineBind = device.createBindGroup({
      layout: this.linePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.viewBuf } },
        { binding: 1, resource: { buffer: this.xBuf } },
        { binding: 2, resource: { buffer: this.yBuf } },
        { binding: 3, resource: { buffer: this.metaBuf } },
      ],
    });

    this.hoverBind = device.createBindGroup({
      layout: this.hoverPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.viewBuf } },
        { binding: 1, resource: { buffer: this.xBuf } },
        { binding: 2, resource: { buffer: this.yBuf } },
        { binding: 3, resource: { buffer: this.hoverBuf } },
        { binding: 4, resource: { buffer: this.hoverResult } },
      ],
    });

    this.extentsBind = device.createBindGroup({
      layout: this.extentsPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.xBuf } },
        { binding: 1, resource: { buffer: this.yBuf } },
        { binding: 2, resource: { buffer: this.extentsResult } },
        { binding: 3, resource: { buffer: this.extentsParamsBuf } },
      ],
    });
  }

  /**
   * Compute xMin/xMax/yMin/yMax on the GPU via a parallel atomicMin/Max
   * reduction and write the result into the SeriesBuffer header. Replaces the
   * O(N) JS loop in the format module.
   */
  async computeExtentsGPU(): Promise<void> {
    // Seed: min slots = encode(+Inf), max slots = encode(-Inf).
    const seed = new Uint32Array([0xFF800000, 0x007FFFFF, 0xFF800000, 0x007FFFFF]);
    this.device.queue.writeBuffer(this.extentsResult, 0, seed);
    this.device.queue.writeBuffer(this.extentsParamsBuf, 0,
      new Uint32Array([this.series.pointCount, this.series.seriesCount, 0, 0]));

    const total = this.series.pointCount * this.series.seriesCount;
    const groups = Math.ceil(total / 64);

    const enc = this.device.createCommandEncoder();
    const cp = enc.beginComputePass();
    cp.setPipeline(this.extentsPipeline);
    cp.setBindGroup(0, this.extentsBind);
    cp.dispatchWorkgroups(groups);
    cp.end();
    enc.copyBufferToBuffer(this.extentsResult, 0, this.extentsReadback, 0, 16);
    this.device.queue.submit([enc.finish()]);

    await this.extentsReadback.mapAsync(GPUMapMode.READ);
    const packed = new Uint32Array(this.extentsReadback.getMappedRange().slice(0));
    this.extentsReadback.unmap();

    this.series.headerF[8]  = decodeOrdered(packed[0]!);
    this.series.headerF[9]  = decodeOrdered(packed[1]!);
    this.series.headerF[10] = decodeOrdered(packed[2]!);
    this.series.headerF[11] = decodeOrdered(packed[3]!);
  }

  private uploadData(): void {
    this.device.queue.writeBuffer(this.xBuf,    0, this.series.x);
    this.device.queue.writeBuffer(this.yBuf,    0, this.series.y);
    this.device.queue.writeBuffer(this.metaBuf, 0, this.series.meta);
  }

  private attachEvents(): void {
    new ResizeObserver(() => this.requestRender()).observe(this.canvas);

    this.canvas.addEventListener('mousedown', (e) => {
      this.dragging = true;
      this.lastMouse = [e.clientX, e.clientY];
      this.canvas.style.cursor = 'grabbing';
    });
    window.addEventListener('mouseup', () => {
      this.dragging = false;
      this.canvas.style.cursor = '';
    });
    window.addEventListener('mousemove', (e) => {
      if (this.dragging) {
        const dx = e.clientX - this.lastMouse[0];
        const dy = e.clientY - this.lastMouse[1];
        this.lastMouse = [e.clientX, e.clientY];
        this.panByCssPixels(dx, dy);
      }
    });
    this.canvas.addEventListener('mousemove', (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const px = (e.clientX - rect.left) * devicePixelRatio;
      const py = (e.clientY - rect.top)  * devicePixelRatio;
      this.queueHover(px, py);
    });
    this.canvas.addEventListener('mouseleave', () => {
      if (this.lastHover !== null) {
        this.lastHover = null;
        this.opts.onHover?.(null);
      }
    });
    this.canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      const rect = this.canvas.getBoundingClientRect();
      const factor = Math.exp(-e.deltaY * 0.0015);
      this.zoomAt(e.clientX - rect.left, e.clientY - rect.top, factor);
    }, { passive: false });
    this.canvas.addEventListener('dblclick', () => this.resetView());
  }

  // ---- view math -----------------------------------------------------------

  private panByCssPixels(dxCss: number, dyCss: number): void {
    const wCss = this.canvas.clientWidth || 1;
    const hCss = this.canvas.clientHeight || 1;
    const dx = (dxCss / wCss) * (this.dataMaxX - this.dataMinX);
    const dy = (dyCss / hCss) * (this.dataMaxY - this.dataMinY);
    this.dataMinX -= dx; this.dataMaxX -= dx;
    this.dataMinY += dy; this.dataMaxY += dy;
    this.requestRender();
  }

  private zoomAt(cssX: number, cssY: number, factor: number): void {
    const fx = cssX / (this.canvas.clientWidth || 1);
    const fy = 1 - cssY / (this.canvas.clientHeight || 1);
    const ax = this.dataMinX + fx * (this.dataMaxX - this.dataMinX);
    const ay = this.dataMinY + fy * (this.dataMaxY - this.dataMinY);
    this.dataMinX = ax + (this.dataMinX - ax) / factor;
    this.dataMaxX = ax + (this.dataMaxX - ax) / factor;
    this.dataMinY = ay + (this.dataMinY - ay) / factor;
    this.dataMaxY = ay + (this.dataMaxY - ay) / factor;
    this.requestRender();
  }

  /** Build the View uniform: ndc = (data - offset) * scale. */
  private buildViewUniform(width: number, height: number): ArrayBuffer {
    const buf = new ArrayBuffer(VIEW_BYTES);
    const f = new Float32Array(buf);
    const u = new Uint32Array(buf);
    const cx = (this.dataMinX + this.dataMaxX) * 0.5;
    const cy = (this.dataMinY + this.dataMaxY) * 0.5;
    const sx = 2 / (this.dataMaxX - this.dataMinX);
    // Y is flipped so positive data goes up on screen.
    const sy = 2 / (this.dataMaxY - this.dataMinY);
    f[0] = sx; f[1] = sy;
    f[2] = cx; f[3] = cy;
    f[4] = width; f[5] = height;
    u[6] = this.series.pointCount;
    u[7] = this.series.seriesCount;
    return buf;
  }

  private buildGridUniform(): ArrayBuffer {
    const buf = new ArrayBuffer(GRID_BYTES);
    const f = new Float32Array(buf);
    const stepX = niceStep(this.dataMaxX - this.dataMinX, this.canvas.clientWidth, 100);
    const stepY = niceStep(this.dataMaxY - this.dataMinY, this.canvas.clientHeight, 80);
    f[0] = stepX; f[1] = stepY;
    // 8 bytes pad before vec4 alignment
    f[4] = this.opts.gridColor[0];
    f[5] = this.opts.gridColor[1];
    f[6] = this.opts.gridColor[2];
    f[7] = this.opts.gridColor[3];
    f[8]  = this.opts.background[0];
    f[9]  = this.opts.background[1];
    f[10] = this.opts.background[2];
    f[11] = this.opts.background[3];
    return buf;
  }

  // ---- render --------------------------------------------------------------

  private render(): void {
    const dpr = devicePixelRatio || 1;
    const w = Math.max(1, Math.floor(this.canvas.clientWidth  * dpr));
    const h = Math.max(1, Math.floor(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== w)  this.canvas.width  = w;
    if (this.canvas.height !== h) this.canvas.height = h;

    this.device.queue.writeBuffer(this.viewBuf, 0, this.buildViewUniform(w, h));
    this.device.queue.writeBuffer(this.gridBuf, 0, this.buildGridUniform());

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.context.getCurrentTexture().createView(),
        clearValue: this.opts.background as [number, number, number, number],
        loadOp: 'clear',
        storeOp: 'store',
      }],
    });

    pass.setPipeline(this.gridPipeline);
    pass.setBindGroup(0, this.gridBind);
    pass.draw(3);

    pass.setPipeline(this.linePipeline);
    pass.setBindGroup(0, this.lineBind);
    const segments = (this.series.pointCount - 1) * this.series.seriesCount;
    if (segments > 0) pass.draw(6, segments);
    pass.end();

    this.device.queue.submit([encoder.finish()]);
  }

  // ---- hover ---------------------------------------------------------------

  private queueHover(px: number, py: number): void {
    if (this.hoverPending) return;
    this.hoverPending = true;
    void this.dispatchHover(px, py).finally(() => { this.hoverPending = false; });
  }

  private async dispatchHover(px: number, py: number): Promise<void> {
    const dpr = devicePixelRatio || 1;
    const w = Math.max(1, Math.floor(this.canvas.clientWidth  * dpr));
    const h = Math.max(1, Math.floor(this.canvas.clientHeight * dpr));

    // Reuse view uniform from last frame; rebuild here to be safe (cheap).
    this.device.queue.writeBuffer(this.viewBuf, 0, this.buildViewUniform(w, h));

    const hoverData = new Float32Array(HOVER_BYTES / 4);
    hoverData[0] = px; hoverData[1] = py;
    this.device.queue.writeBuffer(this.hoverBuf, 0, hoverData);
    // Sentinel — atomicMin starts at "max u32" so the first thread always wins.
    this.device.queue.writeBuffer(this.hoverResult, 0, new Uint32Array([0xFFFFFFFF]));

    const total = this.series.pointCount * this.series.seriesCount;
    const groups = Math.ceil(total / 64);

    const enc = this.device.createCommandEncoder();
    const cp = enc.beginComputePass();
    cp.setPipeline(this.hoverPipeline);
    cp.setBindGroup(0, this.hoverBind);
    cp.dispatchWorkgroups(groups);
    cp.end();
    enc.copyBufferToBuffer(this.hoverResult, 0, this.hoverReadback, 0, 4);
    this.device.queue.submit([enc.finish()]);

    await this.hoverReadback.mapAsync(GPUMapMode.READ);
    const packed = new Uint32Array(this.hoverReadback.getMappedRange().slice(0))[0]!;
    this.hoverReadback.unmap();

    const distQ = packed >>> 20;
    const idx   = packed & 0xFFFFF;
    const distPx = distQ;

    if (packed === 0xFFFFFFFF || distPx > this.opts.hoverThresholdPx * dpr) {
      if (this.lastHover !== null) {
        this.lastHover = null;
        this.opts.onHover?.(null);
      }
      return;
    }

    const series = Math.floor(idx / this.series.pointCount);
    const point  = idx - series * this.series.pointCount;
    const info: HoverInfo = {
      series, point,
      x: this.series.x[point]!,
      y: this.series.y[series * this.series.pointCount + point]!,
      distancePx: distPx / dpr,
    };
    if (
      this.lastHover === null ||
      this.lastHover.series !== info.series ||
      this.lastHover.point !== info.point
    ) {
      this.lastHover = info;
      this.opts.onHover?.(info);
    }
  }
}

// Inverse of the f32 → ordered-u32 mapping in extents.wgsl.
const ORDERED_U32 = new Uint32Array(1);
const ORDERED_F32 = new Float32Array(ORDERED_U32.buffer);
function decodeOrdered(u: number): number {
  // Top bit set => was a positive (or +0); flip the top bit back.
  // Top bit clear => was a negative; bitwise-NOT to undo.
  ORDERED_U32[0] = (u & 0x80000000) !== 0 ? (u ^ 0x80000000) >>> 0 : (~u) >>> 0;
  return ORDERED_F32[0]!;
}

async function fetchShader(device: GPUDevice, path: string, label: string): Promise<GPUShaderModule> {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Failed to load shader ${path}: ${res.status}`);
  const code = await res.text();
  return device.createShaderModule({ label, code });
}

// Choose a "nice" tick spacing — round up to 1/2/5 × 10^k so we get roughly the
// requested number of gridlines per axis.
function niceStep(range: number, viewportPx: number, targetPx: number): number {
  if (range <= 0 || viewportPx <= 0) return 1;
  const targetCount = Math.max(2, viewportPx / targetPx);
  const raw = range / targetCount;
  const exp = Math.floor(Math.log10(raw));
  const base = raw / Math.pow(10, exp);
  let nice: number;
  if      (base < 1.5) nice = 1;
  else if (base < 3.5) nice = 2;
  else if (base < 7.5) nice = 5;
  else                 nice = 10;
  return nice * Math.pow(10, exp);
}
