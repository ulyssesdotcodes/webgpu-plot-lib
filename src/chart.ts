import {
  type SeriesBufferView,
  getExtents,
  getAxisColor,
  getAxisCpuExtents,
  getSeriesAxis,
} from './format.js';

// View uniform layout — must match lines.wgsl and grid.wgsl:
//   offset 0   vec2f scale       ndc = (data - offset) * scale  (scale.y unused; Y handled per-axis)
//   offset 8   vec2f offset
//   offset 16  vec2f viewport    physical pixels
//   offset 24  vec2f gridStep    x tick spacing (data units); gridStep.y unused
//   offset 32  u32   pointCount
//   offset 36  u32   seriesCount
const VIEW_BYTES = 40;

// Pixels reserved per axis for tick labels: one slot on the left (axis 0) and one
// per additional axis stacked on the right.
const PX_PER_AXIS = 55;

// Must match the SUBDIVS constant in lines.wgsl and spline.wgsl.
const SUBDIVS = 4;

// SplineParams uniform: 4 × u32 = 16 bytes.
const SPLINE_PARAMS_BYTES = 16;

// GridStyle uniform: vec4 color + vec4 bg = 32 bytes; written once at init.
const STYLE_BYTES = 32;

// ExtentParams uniform: xMin, xMax (f32) + seriesCount, pointCount (u32) = 16 bytes.
const EXTENT_PARAMS_BYTES = 16;

// GPU axes buffer: 8 floats per axis = 32 bytes.
// Layout: yScale, yOffset, _pad, _pad, rgba (color for tick labels).
const AXIS_GPU_BYTES = 32;

// Characters supported by the font atlas (order determines charMeta index).
// '█' (U+2588 FULL BLOCK) is used for the per-series color indicators.
const TEXT_CHARS      = '0123456789.-█';
const TEXT_MAX_CHARS  = 512;
// Floats per CharInst: ndcPos(2) + ndcSize(2) + uvMin(2) + uvMax(2) + color(4).
const TEXT_INST_FLOATS = 12;

export interface HoverInfo {
  point: number;
  x:     number;
  ys:    Float32Array;
}

export interface LineChartOptions {
  /** Background fill. */
  background?: [number, number, number, number];
  /** Grid line color. */
  gridColor?:  [number, number, number, number];
  /**
   * Backbuffer scale relative to CSS pixels. Defaults to 1 — on Linux Chrome
   * without Vulkan the compositor is CPU-bound, so full DPR hurts framerate.
   * Pass `devicePixelRatio` for crisp lines when the compositor path is fast.
   */
  pixelRatio?: number;
  /** Called when the hovered X column changes. null = cursor left the chart. */
  onHover?: (info: HoverInfo | null) => void;
}

export class LineChart {
  private canvas:  HTMLCanvasElement;
  private context: GPUCanvasContext;
  private device:  GPUDevice;
  private format:  GPUTextureFormat;

  private series: SeriesBufferView;
  private opts:   Required<Omit<LineChartOptions, 'onHover'>> & Pick<LineChartOptions, 'onHover'>;

  private viewBuf!:  GPUBuffer;
  private styleBuf!: GPUBuffer;
  private xBuf!:     GPUBuffer;
  private yBuf!:     GPUBuffer;
  private metaBuf!:  GPUBuffer;
  private axesBuf!:  GPUBuffer;        // per-axis yScale/yOffset/color for lines shader
  private seriesAxisBuf!: GPUBuffer;   // u32 per series — axis membership for extent shader

  private splinePipeline!:    GPUComputePipeline;
  private extentPipeline!:    GPUComputePipeline;
  private gridPipeline!:      GPURenderPipeline;
  private linePipeline!:      GPURenderPipeline;
  private splineBind!:        GPUBindGroup;
  private extentBind!:        GPUBindGroup;
  private gridBind!:          GPUBindGroup;
  private lineBind!:          GPUBindGroup;
  private splineBuf!:         GPUBuffer;
  private splineParamsBuf!:   GPUBuffer;
  private extentParamsBuf!:   GPUBuffer;
  private extentResultBuf!:   GPUBuffer;
  private extentReadbackBuf!: GPUBuffer;
  private extentInFlight      = false;
  private extentDirty         = false;

  private textPipeline!:   GPURenderPipeline;
  private textBind!:       GPUBindGroup;
  private textInstBuf!:    GPUBuffer;
  private atlasTexture!:   GPUTexture;
  private atlasSampler!:   GPUSampler;
  private charMeta:        Array<{ uMin: number; uMax: number; width: number }> = [];
  private atlasH           = 0;
  private textCharCount    = 0;

  // View state — data-domain X range and per-axis Y ranges.
  private dataMinX = 0; private dataMaxX = 1;
  private axisMinY!: Float32Array;
  private axisMaxY!: Float32Array;

  // CPU staging for GPU axes buffer (8 floats per axis).
  private axesStagingF!: Float32Array<ArrayBuffer>;

  private dragging   = false;
  private lastMouseX = 0;
  private lastHoverPoint = -1;

  // CSS-pixel canvas dimensions, kept by ResizeObserver.
  private cssWidth  = 1;
  private cssHeight = 1;

  // True once resetView() has written the first full uniform; guards partial writes.
  private viewInitialized = false;
  private renderQueued    = false;

  private lastFrameTs     = 0;
  private frameCount      = 0;
  private cpuTotalMs      = 0;
  private intervalTotalMs = 0;

  static async create(
    canvas:  HTMLCanvasElement,
    series:  SeriesBufferView,
    options: LineChartOptions = {},
  ): Promise<LineChart> {
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
    const device  = await adapter?.requestDevice();
    if (!device || !adapter) throw new Error('WebGPU not available');

    const info = adapter.info;
    console.log('[webgpu] adapter:', {
      vendor: info.vendor, architecture: info.architecture,
      device: info.device, description: info.description,
    });

    const context = canvas.getContext('webgpu');
    if (!context) throw new Error('canvas has no webgpu context');

    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format, alphaMode: 'opaque' });

    const [grid, lines, spline, extent, text] = await Promise.all([
      fetchShader(device, '/shaders/grid.wgsl',   'grid'),
      fetchShader(device, '/shaders/lines.wgsl',  'lines'),
      fetchShader(device, '/shaders/spline.wgsl', 'spline'),
      fetchShader(device, '/shaders/extent.wgsl', 'extent'),
      fetchShader(device, '/shaders/text.wgsl',   'text'),
    ]);

    const chart = new LineChart(canvas, context, device, format, series, options);
    chart.buildResources(grid, lines, spline, extent, text);
    chart.attachEvents();
    chart.resetView();
    return chart;
  }

  private constructor(
    canvas:  HTMLCanvasElement,
    context: GPUCanvasContext,
    device:  GPUDevice,
    format:  GPUTextureFormat,
    series:  SeriesBufferView,
    options: LineChartOptions,
  ) {
    this.canvas  = canvas;
    this.context = context;
    this.device  = device;
    this.format  = format;
    this.series  = series;
    this.opts = {
      background: options.background ?? [0.07, 0.08, 0.11, 1],
      gridColor:  options.gridColor  ?? [0.22, 0.24, 0.28, 1],
      pixelRatio: options.pixelRatio ?? 1,
      ...(options.onHover !== undefined ? { onHover: options.onHover } : {}),
    };

    this.axisMinY     = new Float32Array(series.axisCount);
    this.axisMaxY     = new Float32Array(series.axisCount);
    const axesBacking = new ArrayBuffer(series.axisCount * AXIS_GPU_BYTES);
    this.axesStagingF = new Float32Array(axesBacking);
  }

  // ---- public API ----------------------------------------------------------

  resetView(): void {
    const e    = getExtents(this.series);
    const padX = (e.xMax - e.xMin) * 0.02 || 1;
    this.dataMinX = e.xMin - padX;
    this.dataMaxX = e.xMax + padX;

    const axExtents = getAxisCpuExtents(this.series);
    for (let a = 0; a < this.series.axisCount; a++) {
      const ae   = axExtents[a]!;
      const padY = (ae.yMax - ae.yMin) * 0.05 || 1;
      this.axisMinY[a] = ae.yMin - padY;
      this.axisMaxY[a] = ae.yMax + padY;
    }

    const dpr = this.opts.pixelRatio;
    const w   = Math.max(1, Math.floor(this.cssWidth  * dpr));
    const h   = Math.max(1, Math.floor(this.cssHeight * dpr));
    this.uploadAxes();
    this.writeViewAll(w, h);
    this.viewInitialized = true;
    this.updateLabels();
    this.requestAxisExtents();
    this.requestRender();
  }

  setData(series: SeriesBufferView): void {
    if (
      series.seriesCount !== this.series.seriesCount ||
      series.pointCount  !== this.series.pointCount  ||
      series.axisCount   !== this.series.axisCount
    ) {
      throw new Error('setData: series/point/axis shape must match');
    }
    this.series = series;
    this.uploadData();
    this.bakeSpline();
    this.resetView();
  }

  requestRender(): void {
    if (this.renderQueued) return;
    this.renderQueued = true;
    requestAnimationFrame(() => {
      this.renderQueued = false;
      this.render();
    });
  }

  // ---- setup ---------------------------------------------------------------

  private buildResources(gridMod: GPUShaderModule, lineMod: GPUShaderModule, splineMod: GPUShaderModule, extentMod: GPUShaderModule, textMod: GPUShaderModule): void {
    const { device, series } = this;

    const totalSamples      = SUBDIVS * (series.pointCount - 1) + 1;
    const extentResultBytes = series.axisCount * 8;

    // Data buffers.
    this.viewBuf         = device.createBuffer({ label: 'view',          size: VIEW_BYTES,                              usage: GPUBufferUsage.UNIFORM  | GPUBufferUsage.COPY_DST });
    this.styleBuf        = device.createBuffer({ label: 'style',         size: STYLE_BYTES,                             usage: GPUBufferUsage.UNIFORM  | GPUBufferUsage.COPY_DST });
    this.xBuf            = device.createBuffer({ label: 'x',             size: series.x.byteLength,                     usage: GPUBufferUsage.STORAGE  | GPUBufferUsage.COPY_DST });
    this.yBuf            = device.createBuffer({ label: 'y',             size: series.y.byteLength,                     usage: GPUBufferUsage.STORAGE  | GPUBufferUsage.COPY_DST });
    this.metaBuf         = device.createBuffer({ label: 'meta',          size: series.meta.byteLength,                  usage: GPUBufferUsage.STORAGE  | GPUBufferUsage.COPY_DST });
    this.axesBuf         = device.createBuffer({ label: 'axes',          size: series.axisCount * AXIS_GPU_BYTES,       usage: GPUBufferUsage.STORAGE  | GPUBufferUsage.COPY_DST });
    this.seriesAxisBuf   = device.createBuffer({ label: 'series-axis',   size: series.seriesCount * 4,                  usage: GPUBufferUsage.STORAGE  | GPUBufferUsage.COPY_DST });
    this.splineParamsBuf = device.createBuffer({ label: 'spline-params', size: SPLINE_PARAMS_BYTES,                     usage: GPUBufferUsage.UNIFORM  | GPUBufferUsage.COPY_DST });
    this.splineBuf          = device.createBuffer({ label: 'spline',          size: series.seriesCount * totalSamples * 8,  usage: GPUBufferUsage.STORAGE });
    this.extentParamsBuf    = device.createBuffer({ label: 'extent-params',   size: EXTENT_PARAMS_BYTES,                    usage: GPUBufferUsage.UNIFORM  | GPUBufferUsage.COPY_DST });
    this.extentResultBuf    = device.createBuffer({ label: 'extent-result',   size: extentResultBytes,                      usage: GPUBufferUsage.STORAGE  | GPUBufferUsage.COPY_SRC });
    this.extentReadbackBuf  = device.createBuffer({ label: 'extent-readback', size: extentResultBytes,                      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

    this.uploadData();

    // Grid style uniform — written once at init, never changes.
    const styleData = new Float32Array(STYLE_BYTES / 4);
    styleData.set(this.opts.gridColor,  0);
    styleData.set(this.opts.background, 4);
    device.queue.writeBuffer(this.styleBuf, 0, styleData);

    // Pipelines.
    this.splinePipeline = device.createComputePipeline({
      label:  'spline',
      layout: 'auto',
      compute: { module: splineMod, entryPoint: 'main' },
    });

    this.extentPipeline = device.createComputePipeline({
      label:  'extent',
      layout: 'auto',
      compute: { module: extentMod, entryPoint: 'main' },
    });

    this.gridPipeline = device.createRenderPipeline({
      label:  'grid',
      layout: 'auto',
      vertex:   { module: gridMod, entryPoint: 'vs' },
      fragment: { module: gridMod, entryPoint: 'fs', targets: [{ format: this.format }] },
      primitive: { topology: 'triangle-list' },
    });

    this.linePipeline = device.createRenderPipeline({
      label:  'lines',
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
      primitive: { topology: 'triangle-strip' },
    });

    // Bind groups.
    this.splineBind = device.createBindGroup({
      layout: this.splinePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.splineParamsBuf } },
        { binding: 1, resource: { buffer: this.xBuf            } },
        { binding: 2, resource: { buffer: this.yBuf            } },
        { binding: 3, resource: { buffer: this.splineBuf       } },
      ],
    });

    this.gridBind = device.createBindGroup({
      layout: this.gridPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.viewBuf   } },
        { binding: 1, resource: { buffer: this.styleBuf  } },
      ],
    });

    this.extentBind = device.createBindGroup({
      layout: this.extentPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.extentParamsBuf  } },
        { binding: 1, resource: { buffer: this.xBuf             } },
        { binding: 2, resource: { buffer: this.yBuf             } },
        { binding: 3, resource: { buffer: this.extentResultBuf  } },
        { binding: 4, resource: { buffer: this.seriesAxisBuf    } },
      ],
    });

    this.lineBind = device.createBindGroup({
      layout: this.linePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.viewBuf   } },
        { binding: 1, resource: { buffer: this.splineBuf } },
        { binding: 2, resource: { buffer: this.metaBuf   } },
        { binding: 3, resource: { buffer: this.axesBuf   } },
      ],
    });

    // Font atlas + text pipeline.
    this.buildAtlas();

    this.textInstBuf = device.createBuffer({
      label: 'text-inst',
      size:  TEXT_MAX_CHARS * TEXT_INST_FLOATS * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.textPipeline = device.createRenderPipeline({
      label:  'text',
      layout: 'auto',
      vertex:   { module: textMod, entryPoint: 'vs' },
      fragment: {
        module: textMod, entryPoint: 'fs',
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

    this.textBind = device.createBindGroup({
      layout: this.textPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.textInstBuf  } },
        { binding: 1, resource: this.atlasTexture.createView() },
        { binding: 2, resource: this.atlasSampler             },
      ],
    });

    // Bake the initial spline (pipelines and bind groups are ready now).
    this.bakeSpline();
  }

  private buildAtlas(): void {
    const font  = '12px monospace';
    const pad   = 1;

    const mCanvas = document.createElement('canvas');
    mCanvas.width  = 1;
    mCanvas.height = 1;
    const mctx = mCanvas.getContext('2d')!;
    mctx.font = font;

    const metrics  = mctx.measureText('0');
    this.atlasH    = Math.ceil(metrics.actualBoundingBoxAscent + metrics.actualBoundingBoxDescent) + 4;

    const slotWidths = Array.from(TEXT_CHARS).map(c => Math.ceil(mctx.measureText(c).width) + pad * 2);
    const totalW     = slotWidths.reduce((a, b) => a + b, 0);
    const texW       = Math.ceil(totalW / 64) * 64;

    const aCanvas = document.createElement('canvas');
    aCanvas.width  = texW;
    aCanvas.height = this.atlasH;
    const ctx = aCanvas.getContext('2d')!;
    ctx.font         = font;
    ctx.fillStyle    = 'white';
    ctx.textBaseline = 'middle';

    let x = 0;
    this.charMeta = [];
    for (let i = 0; i < TEXT_CHARS.length; i++) {
      const w = slotWidths[i]!;
      ctx.fillText(TEXT_CHARS[i]!, x + pad, this.atlasH / 2);
      this.charMeta.push({ uMin: x / texW, uMax: (x + w) / texW, width: w });
      x += w;
    }

    const imgData = ctx.getImageData(0, 0, texW, this.atlasH);

    this.atlasTexture = this.device.createTexture({
      label:  'font-atlas',
      size:   [texW, this.atlasH],
      format: 'rgba8unorm',
      usage:  GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    this.device.queue.writeTexture(
      { texture: this.atlasTexture },
      imgData.data,
      { bytesPerRow: texW * 4, rowsPerImage: this.atlasH },
      [texW, this.atlasH],
    );

    this.atlasSampler = this.device.createSampler({
      label:     'font-sampler',
      magFilter: 'linear',
      minFilter: 'linear',
    });
  }

  private bakeSpline(): void {
    const totalSamples = SUBDIVS * (this.series.pointCount - 1) + 1;
    const params = new Uint32Array(4);
    params[0] = this.series.pointCount;
    params[1] = this.series.seriesCount;
    params[2] = totalSamples;
    this.device.queue.writeBuffer(this.splineParamsBuf, 0, params);

    const totalWork = this.series.seriesCount * totalSamples;
    const groups    = Math.ceil(totalWork / 64);

    const enc = this.device.createCommandEncoder();
    const cp  = enc.beginComputePass();
    cp.setPipeline(this.splinePipeline);
    cp.setBindGroup(0, this.splineBind);
    cp.dispatchWorkgroups(groups);
    cp.end();
    this.device.queue.submit([enc.finish()]);
  }

  private uploadData(): void {
    this.device.queue.writeBuffer(this.xBuf,    0, this.series.x);
    this.device.queue.writeBuffer(this.yBuf,    0, this.series.y);
    this.device.queue.writeBuffer(this.metaBuf, 0, this.series.meta);
    this.uploadSeriesAxis();
  }

  private uploadSeriesAxis(): void {
    const buf = new Uint32Array(this.series.seriesCount);
    for (let s = 0; s < this.series.seriesCount; s++) {
      buf[s] = getSeriesAxis(this.series, s);
    }
    this.device.queue.writeBuffer(this.seriesAxisBuf, 0, buf);
  }

  // Upload the per-axis GPU buffer (yScale, yOffset, color) from current axisMinY/axisMaxY.
  private uploadAxes(): void {
    for (let a = 0; a < this.series.axisCount; a++) {
      const yMin  = this.axisMinY[a]!;
      const yMax  = this.axisMaxY[a]!;
      const range = yMax - yMin || 1;
      const o = a * 8;
      this.axesStagingF[o + 0] = 2 / range;                  // yScale
      this.axesStagingF[o + 1] = (yMin + yMax) * 0.5;        // yOffset
      this.axesStagingF[o + 2] = 0;                           // _pad
      this.axesStagingF[o + 3] = 0;                           // _pad
      const c = getAxisColor(this.series, a);
      this.axesStagingF[o + 4] = c[0];
      this.axesStagingF[o + 5] = c[1];
      this.axesStagingF[o + 6] = c[2];
      this.axesStagingF[o + 7] = c[3];
    }
    this.device.queue.writeBuffer(this.axesBuf, 0, this.axesStagingF);
  }

  private attachEvents(): void {
    const sync = () => {
      this.cssWidth  = this.canvas.clientWidth  || 1;
      this.cssHeight = this.canvas.clientHeight || 1;
      if (this.viewInitialized) {
        const dpr = this.opts.pixelRatio;
        const w   = Math.max(1, Math.floor(this.cssWidth  * dpr));
        const h   = Math.max(1, Math.floor(this.cssHeight * dpr));
        this.writeViewAll(w, h);
      }
      this.updateLabels();
      this.requestRender();
    };
    sync();
    new ResizeObserver(sync).observe(this.canvas);

    this.canvas.addEventListener('mousedown', (e) => {
      this.dragging   = true;
      this.lastMouseX = e.clientX;
      this.canvas.style.cursor = 'grabbing';
    });
    window.addEventListener('mouseup', () => {
      if (this.dragging) this.requestAxisExtents();
      this.dragging = false;
      this.canvas.style.cursor = '';
    });
    window.addEventListener('mousemove', (e) => {
      if (!this.dragging) return;
      const dx = e.clientX - this.lastMouseX;
      this.lastMouseX = e.clientX;
      this.panByCssPixels(dx);
    });
    this.canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      const rect   = this.canvas.getBoundingClientRect();
      const factor = Math.exp(-e.deltaY * 0.0015);
      this.zoomAt(e.clientX - rect.left, factor);
    }, { passive: false });
    this.canvas.addEventListener('dblclick', () => this.resetView());

    this.canvas.addEventListener('mousemove', (e) => {
      if (this.dragging) return;
      const rect = this.canvas.getBoundingClientRect();
      this.pickX(e.clientX - rect.left);
    });
    this.canvas.addEventListener('mouseleave', () => {
      if (this.lastHoverPoint !== -1) {
        this.lastHoverPoint = -1;
        this.opts.onHover?.(null);
      }
    });
  }

  private pickX(cssX: number): void {
    const leftPx  = this.gutterLeft();
    const rightPx = this.gutterRight();
    const plotW   = Math.max(1, this.cssWidth - leftPx - rightPx);
    const dataX   = this.dataMinX + (Math.max(0, cssX - leftPx) / plotW) * (this.dataMaxX - this.dataMinX);
    const x = this.series.x;
    const N = this.series.pointCount;

    let lo = 0, hi = N;
    while (lo < hi) { const m = (lo + hi) >>> 1; if (x[m]! < dataX) lo = m + 1; else hi = m; }
    const i = (lo > 0 && lo < N && Math.abs(x[lo - 1]! - dataX) <= Math.abs(x[lo]! - dataX))
      ? lo - 1
      : Math.min(lo, N - 1);

    if (i === this.lastHoverPoint) return;
    this.lastHoverPoint = i;

    const ys = new Float32Array(this.series.seriesCount);
    for (let s = 0; s < this.series.seriesCount; s++) {
      ys[s] = this.series.y[s * N + i]!;
    }
    this.opts.onHover?.({ point: i, x: x[i]!, ys });
  }

  // ---- view math -----------------------------------------------------------

  // All axes stack on the left; no right gutter.
  private gutterLeft():  number { return this.series.axisCount * PX_PER_AXIS; }
  private gutterRight(): number { return 0; }

  // Gutter-aware X scale/offset: the data range maps to the plot sub-rectangle, not the
  // full canvas. By folding the gutter into scale/offset, no shader changes are needed.
  private computeXScaleOffset(): { scaleX: number; offsetX: number } {
    const w      = this.cssWidth;
    const left   = this.gutterLeft();
    const right  = this.gutterRight();
    const plotW  = Math.max(1, w - left - right);
    const range  = this.dataMaxX - this.dataMinX || 1;
    const scaleX = (plotW / w) * 2 / range;
    const centre = (this.dataMinX + this.dataMaxX) * 0.5;
    // Canvas-NDC centre of the plot area.
    const plotCentreNDC = 2 * (left + plotW * 0.5) / w - 1;
    // Absorb the NDC translation into offset so the shader formula (pos-offset)*scale is unchanged.
    const offsetX = centre - plotCentreNDC / scaleX;
    return { scaleX, offsetX };
  }

  private panByCssPixels(dxCss: number): void {
    const leftPx  = this.gutterLeft();
    const rightPx = this.gutterRight();
    const plotW   = Math.max(1, this.cssWidth - leftPx - rightPx);
    const dx = (dxCss / plotW) * (this.dataMaxX - this.dataMinX);
    this.dataMinX -= dx; this.dataMaxX -= dx;
    this.writeViewOffset();
    this.updateLabels();
    this.requestRender();
  }

  private zoomAt(cssX: number, factor: number): void {
    const leftPx  = this.gutterLeft();
    const rightPx = this.gutterRight();
    const plotW   = Math.max(1, this.cssWidth - leftPx - rightPx);
    const fx = Math.max(0, cssX - leftPx) / plotW;
    const ax = this.dataMinX + fx * (this.dataMaxX - this.dataMinX);
    this.dataMinX = ax + (this.dataMinX - ax) / factor;
    this.dataMaxX = ax + (this.dataMaxX - ax) / factor;
    this.writeViewScaleOffsetStep();
    this.updateLabels();
    this.requestAxisExtents();
    this.requestRender();
  }

  // ---- uniform helpers -----------------------------------------------------

  private viewStaging  = new ArrayBuffer(VIEW_BYTES);
  private viewStagingF = new Float32Array(this.viewStaging);
  private viewStagingU = new Uint32Array(this.viewStaging);

  private extentParams  = new ArrayBuffer(EXTENT_PARAMS_BYTES);
  private extentParamsF = new Float32Array(this.extentParams);
  private extentParamsU = new Uint32Array(this.extentParams);

  // Drag: only the X offset changes.
  private writeViewOffset(): void {
    const { offsetX } = this.computeXScaleOffset();
    const f = this.viewStagingF;
    f[2] = offsetX;
    f[3] = 0; // scale.y/offset.y unused (per-axis Y handled in axesBuf)
    this.device.queue.writeBuffer(this.viewBuf, 8, this.viewStaging, 8, 8);
  }

  // Zoom: scale, offset, and gridStep change; viewport stays.
  private writeViewScaleOffsetStep(): void {
    const { scaleX, offsetX } = this.computeXScaleOffset();
    const f = this.viewStagingF;
    f[0] = scaleX;
    f[1] = 0; // scale.y unused
    f[2] = offsetX;
    f[3] = 0; // offset.y unused
    f[6] = niceStep(this.dataMaxX - this.dataMinX, this.cssWidth, 100);
    f[7] = 0; // gridStep.y unused (no Y grid lines)
    this.device.queue.writeBuffer(this.viewBuf, 0, this.viewStaging, 0, 32);
  }

  // Init / reset / resize: write the full 40-byte struct.
  private writeViewAll(w: number, h: number): void {
    const { scaleX, offsetX } = this.computeXScaleOffset();
    const f = this.viewStagingF;
    const u = this.viewStagingU;
    f[0] = scaleX;
    f[1] = 0;
    f[2] = offsetX;
    f[3] = 0;
    f[4] = w; f[5] = h;
    f[6] = niceStep(this.dataMaxX - this.dataMinX, this.cssWidth, 100);
    f[7] = 0;
    u[8] = this.series.pointCount;
    u[9] = this.series.seriesCount;
    this.device.queue.writeBuffer(this.viewBuf, 0, this.viewStaging);
  }

  // ---- extent compute + Y autoscale ----------------------------------------

  private requestAxisExtents(): void {
    if (this.extentInFlight) { this.extentDirty = true; return; }
    this.extentInFlight = true;
    this.extentDirty    = false;

    this.extentParamsF[0] = this.dataMinX;
    this.extentParamsF[1] = this.dataMaxX;
    this.extentParamsU[2] = this.series.seriesCount;
    this.extentParamsU[3] = this.series.pointCount;
    this.device.queue.writeBuffer(this.extentParamsBuf, 0, this.extentParams);

    const resultBytes = this.series.axisCount * 8;
    const enc = this.device.createCommandEncoder();
    const cp  = enc.beginComputePass();
    cp.setPipeline(this.extentPipeline);
    cp.setBindGroup(0, this.extentBind);
    cp.dispatchWorkgroups(this.series.axisCount); // one workgroup per axis
    cp.end();
    enc.copyBufferToBuffer(this.extentResultBuf, 0, this.extentReadbackBuf, 0, resultBytes);
    this.device.queue.submit([enc.finish()]);

    this.extentReadbackBuf.mapAsync(GPUMapMode.READ).then(() => {
      const arr = new Float32Array(this.extentReadbackBuf.getMappedRange());
      const results = arr.slice(); // copy before unmap
      this.extentReadbackBuf.unmap();
      this.extentInFlight = false;
      for (let a = 0; a < this.series.axisCount; a++) {
        this.applyAxisAutoscale(a, results[a * 2]!, results[a * 2 + 1]!);
      }
      if (this.extentDirty) this.requestAxisExtents();
    }).catch(() => {
      this.extentInFlight = false;
    });
  }

  private applyAxisAutoscale(a: number, yMin: number, yMax: number): void {
    if (!isFinite(yMin) || !isFinite(yMax) || yMax <= yMin) return;
    const pad = (yMax - yMin) * 0.05 || 0.5;
    this.axisMinY[a] = yMin - pad;
    this.axisMaxY[a] = yMax + pad;
    this.uploadAxes();
    this.updateLabels();
    this.requestRender();
  }

  // ---- label overlay -------------------------------------------------------

  private updateLabels(): void {
    if (!this.viewInitialized || this.charMeta.length === 0) return;

    const w     = this.cssWidth;
    const h     = this.cssHeight;
    const stepX = this.viewStagingF[6]!;

    const leftPx  = this.gutterLeft();
    const rightPx = this.gutterRight();
    const plotW   = Math.max(1, w - leftPx - rightPx);

    const inst = new Float32Array(TEXT_MAX_CHARS * TEXT_INST_FLOATS);
    let count  = 0;

    const emitChar = (c: string, curX: number, topY: number, color: readonly [number, number, number, number]): number => {
      if (count >= TEXT_MAX_CHARS) return 0;
      const idx = TEXT_CHARS.indexOf(c);
      if (idx < 0) return 0;
      const m    = this.charMeta[idx]!;
      const ndcX =  (curX / w) * 2 - 1;
      const ndcY = 1 - (topY / h) * 2;
      const ndcW = (m.width / w) * 2;
      const ndcH = (this.atlasH / h) * 2;
      const off  = count * TEXT_INST_FLOATS;
      inst[off]    = ndcX;     inst[off+1]  = ndcY;
      inst[off+2]  = ndcW;     inst[off+3]  = ndcH;
      inst[off+4]  = m.uMin;   inst[off+5]  = 0;
      inst[off+6]  = m.uMax;   inst[off+7]  = 1;
      inst[off+8]  = color[0]; inst[off+9]  = color[1];
      inst[off+10] = color[2]; inst[off+11] = color[3];
      count++;
      return m.width;
    };

    const emitLabel = (text: string, anchorX: number, anchorY: number, centerX: boolean, color: readonly [number, number, number, number]) => {
      let totalW = 0;
      for (const c of text) {
        const idx = TEXT_CHARS.indexOf(c);
        if (idx >= 0) totalW += this.charMeta[idx]!.width;
      }
      let curX = centerX ? anchorX - totalW / 2 : anchorX;
      const topY = anchorY - this.atlasH / 2;
      for (const c of text) curX += emitChar(c, curX, topY, color);
    };

    const white = [1, 1, 1, 1] as const;

    // X ticks — positioned within the plot area.
    if (stepX > 0 && plotW > 0) {
      const first = Math.ceil(this.dataMinX / stepX) * stepX;
      for (let v = first; v < this.dataMaxX + stepX * 1e-6; v += stepX) {
        const px = leftPx + (v - this.dataMinX) / (this.dataMaxX - this.dataMinX) * plotW;
        if (px < leftPx || px > w - rightPx) continue;
        emitLabel(formatTick(v, stepX), px, h - this.atlasH / 2 - 4, true, white);
      }
    }

    // Y ticks — one column per axis, all on the left.
    // Axis 0 is innermost (closest to plot); axis N-1 is outermost.
    for (let a = 0; a < this.series.axisCount; a++) {
      const yMin   = this.axisMinY[a]!;
      const yMax   = this.axisMaxY[a]!;
      const yRange = yMax - yMin;
      if (yRange <= 0) continue;

      const stepY = niceStep(yRange, h, 80);
      if (stepY <= 0) continue;

      // Column for axis a: axis 0 is innermost (rightmost of the left gutters).
      const colLeft = (this.series.axisCount - 1 - a) * PX_PER_AXIS;
      const labelX  = colLeft + 4;

      // Tick labels in white.
      const first = Math.ceil(yMin / stepY) * stepY;
      for (let v = first; v < yMax + stepY * 1e-6; v += stepY) {
        const py = (1 - (v - yMin) / yRange) * h;
        if (py < 0 || py > h) continue;
        emitLabel(formatTick(v, stepY), labelX, py, false, white);
      }

      // Colored squares at the top of the column — one per series on this axis.
      const sqIdx = TEXT_CHARS.indexOf('█');
      const sqW   = sqIdx >= 0 ? this.charMeta[sqIdx]!.width : 8;
      const colCenterX = colLeft + PX_PER_AXIS / 2;
      const srcs: number[] = [];
      for (let s = 0; s < this.series.seriesCount; s++) {
        if (getSeriesAxis(this.series, s) === a) srcs.push(s);
      }
      let sqX = colCenterX - (srcs.length * sqW) / 2;
      for (const s of srcs) {
        const color = [
          this.series.meta[s * 8 + 0]!,
          this.series.meta[s * 8 + 1]!,
          this.series.meta[s * 8 + 2]!,
          1,
        ] as const;
        emitChar('█', sqX, 4, color);
        sqX += sqW;
      }
    }

    this.textCharCount = count;
    if (count > 0) {
      this.device.queue.writeBuffer(this.textInstBuf, 0, inst, 0, count * TEXT_INST_FLOATS);
    }
  }

  // ---- visible sample range ------------------------------------------------

  private visibleSamples(): [number, number] {
    const x   = this.series.x;
    const N   = this.series.pointCount;
    const tot = SUBDIVS * (N - 1) + 1;

    let lo = 0, hi = N;
    while (lo < hi) { const m = (lo + hi) >>> 1; if (x[m]! < this.dataMinX) lo = m + 1; else hi = m; }
    const firstPt = lo;

    lo = 0; hi = N;
    while (lo < hi) { const m = (lo + hi) >>> 1; if (x[m]! <= this.dataMaxX) lo = m + 1; else hi = m; }
    const lastPt = lo - 1;

    const first = Math.max(0,       firstPt * SUBDIVS - SUBDIVS);
    const last  = Math.min(tot - 1, lastPt  * SUBDIVS + SUBDIVS);
    return [first, last];
  }

  // ---- render --------------------------------------------------------------

  private render(): void {
    const t0       = performance.now();
    const interval = this.lastFrameTs > 0 ? t0 - this.lastFrameTs : 0;
    this.lastFrameTs = t0;

    const dpr = this.opts.pixelRatio;
    const w   = Math.max(1, Math.floor(this.cssWidth  * dpr));
    const h   = Math.max(1, Math.floor(this.cssHeight * dpr));
    if (this.canvas.width  !== w) this.canvas.width  = w;
    if (this.canvas.height !== h) this.canvas.height = h;

    const encoder = this.device.createCommandEncoder();
    const pass    = encoder.beginRenderPass({
      colorAttachments: [{
        view:       this.context.getCurrentTexture().createView(),
        clearValue: this.opts.background,
        loadOp:     'clear',
        storeOp:    'store',
      }],
    });

    pass.setPipeline(this.gridPipeline);
    pass.setBindGroup(0, this.gridBind);
    pass.draw(3);

    pass.setPipeline(this.linePipeline);
    pass.setBindGroup(0, this.lineBind);
    if (this.series.pointCount >= 2) {
      const [firstS, lastS] = this.visibleSamples();
      const count = lastS - firstS + 1;
      if (count > 0) pass.draw(2 * count, this.series.seriesCount, 2 * firstS);
    }

    if (this.textCharCount > 0) {
      pass.setPipeline(this.textPipeline);
      pass.setBindGroup(0, this.textBind);
      pass.draw(6, this.textCharCount);
    }

    pass.end();

    this.device.queue.submit([encoder.finish()]);

    this.cpuTotalMs += performance.now() - t0;
    if (interval > 0) this.intervalTotalMs += interval;
    if (++this.frameCount === 30) {
      const avgCpu      = (this.cpuTotalMs / 30).toFixed(2);
      const avgInterval = (this.intervalTotalMs / 29).toFixed(2);
      console.log(`[render] avg cpu ${avgCpu}ms / avg interval ${avgInterval}ms (${(1000 / Number(avgInterval)).toFixed(1)} fps)`);
      this.frameCount      = 0;
      this.cpuTotalMs      = 0;
      this.intervalTotalMs = 0;
    }
  }
}

async function fetchShader(device: GPUDevice, path: string, label: string): Promise<GPUShaderModule> {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Failed to load shader ${path}: ${res.status}`);
  const code = await res.text();
  return device.createShaderModule({ label, code });
}

function formatTick(v: number, step: number): string {
  const decimals = Math.max(0, -Math.floor(Math.log10(step)));
  return v.toFixed(decimals);
}

function niceStep(range: number, viewportPx: number, targetPx: number): number {
  if (range <= 0 || viewportPx <= 0) return 1;
  const targetCount = Math.max(2, viewportPx / targetPx);
  const raw  = range / targetCount;
  const exp  = Math.floor(Math.log10(raw));
  const base = raw / Math.pow(10, exp);
  let nice: number;
  if      (base < 1.5) nice = 1;
  else if (base < 3.5) nice = 2;
  else if (base < 7.5) nice = 5;
  else                 nice = 10;
  return nice * Math.pow(10, exp);
}
