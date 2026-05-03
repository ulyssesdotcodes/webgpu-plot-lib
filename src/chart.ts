import {
  type SeriesBufferView,
  getExtents,
} from './format.js';

// View uniform layout — must match lines.wgsl and grid.wgsl:
//   offset 0   vec2f scale       ndc = (data - offset) * scale
//   offset 8   vec2f offset      data-domain centre
//   offset 16  vec2f viewport    physical pixels
//   offset 24  vec2f gridStep    nice tick spacing (data units)
//   offset 32  u32   pointCount
//   offset 36  u32   seriesCount
const VIEW_BYTES = 40;

// Must match the SUBDIVS constant in lines.wgsl and spline.wgsl.
const SUBDIVS = 4;

// SplineParams uniform: 4 × u32 = 16 bytes.
const SPLINE_PARAMS_BYTES = 16;

// GridStyle uniform: vec4 color + vec4 bg = 32 bytes; written once at init.
const STYLE_BYTES = 32;

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
}

export class LineChart {
  private canvas:  HTMLCanvasElement;
  private context: GPUCanvasContext;
  private device:  GPUDevice;
  private format:  GPUTextureFormat;

  private series: SeriesBufferView;
  private opts:   Required<LineChartOptions>;

  private viewBuf!:  GPUBuffer;
  private styleBuf!: GPUBuffer;
  private xBuf!:     GPUBuffer;
  private yBuf!:     GPUBuffer;
  private metaBuf!:  GPUBuffer;

  private splinePipeline!:   GPUComputePipeline;
  private gridPipeline!:     GPURenderPipeline;
  private linePipeline!:     GPURenderPipeline;
  private splineBind!:       GPUBindGroup;
  private gridBind!:         GPUBindGroup;
  private lineBind!:         GPUBindGroup;
  private splineBuf!:        GPUBuffer;
  private splineParamsBuf!:  GPUBuffer;


  // View state — data-domain rectangle currently visible.
  private dataMinX = 0; private dataMaxX = 1;
  private dataMinY = 0; private dataMaxY = 1;

  private dragging   = false;
  private lastMouseX = 0;
  private lastMouseY = 0;

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

    const [grid, lines, spline] = await Promise.all([
      fetchShader(device, '/shaders/grid.wgsl',   'grid'),
      fetchShader(device, '/shaders/lines.wgsl',  'lines'),
      fetchShader(device, '/shaders/spline.wgsl', 'spline'),
    ]);

    const chart = new LineChart(canvas, context, device, format, series, options);
    chart.buildResources(grid, lines, spline);
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
    };
  }

  // ---- public API ----------------------------------------------------------

  resetView(): void {
    const e    = getExtents(this.series);
    const padX = (e.xMax - e.xMin) * 0.02 || 1;
    const padY = (e.yMax - e.yMin) * 0.05 || 1;
    this.dataMinX = e.xMin - padX;
    this.dataMaxX = e.xMax + padX;
    this.dataMinY = e.yMin - padY;
    this.dataMaxY = e.yMax + padY;
    const dpr = this.opts.pixelRatio;
    const w   = Math.max(1, Math.floor(this.cssWidth  * dpr));
    const h   = Math.max(1, Math.floor(this.cssHeight * dpr));
    this.writeViewAll(w, h);
    this.viewInitialized = true;
    this.requestRender();
  }

  setData(series: SeriesBufferView): void {
    if (
      series.seriesCount !== this.series.seriesCount ||
      series.pointCount  !== this.series.pointCount
    ) {
      throw new Error('setData: series/point shape must match');
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

  private buildResources(gridMod: GPUShaderModule, lineMod: GPUShaderModule, splineMod: GPUShaderModule): void {
    const { device, series } = this;

    const totalSamples = SUBDIVS * (series.pointCount - 1) + 1;

    // Data buffers.
    this.viewBuf         = device.createBuffer({ label: 'view',          size: VIEW_BYTES,                             usage: GPUBufferUsage.UNIFORM  | GPUBufferUsage.COPY_DST });
    this.styleBuf        = device.createBuffer({ label: 'style',         size: STYLE_BYTES,                            usage: GPUBufferUsage.UNIFORM  | GPUBufferUsage.COPY_DST });
    this.xBuf            = device.createBuffer({ label: 'x',             size: series.x.byteLength,                    usage: GPUBufferUsage.STORAGE  | GPUBufferUsage.COPY_DST });
    this.yBuf            = device.createBuffer({ label: 'y',             size: series.y.byteLength,                    usage: GPUBufferUsage.STORAGE  | GPUBufferUsage.COPY_DST });
    this.metaBuf         = device.createBuffer({ label: 'meta',          size: series.meta.byteLength,                 usage: GPUBufferUsage.STORAGE  | GPUBufferUsage.COPY_DST });
    this.splineParamsBuf = device.createBuffer({ label: 'spline-params', size: SPLINE_PARAMS_BYTES,                    usage: GPUBufferUsage.UNIFORM  | GPUBufferUsage.COPY_DST });
    this.splineBuf       = device.createBuffer({ label: 'spline',        size: series.seriesCount * totalSamples * 8,  usage: GPUBufferUsage.STORAGE });

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

    // this.gridBind = device.createBindGroup({
    //   layout: this.gridPipeline.getBindGroupLayout(0),
    //   entries: [
    //     { binding: 0, resource: { buffer: this.viewBuf   } },
    //     { binding: 1, resource: { buffer: this.styleBuf  } },
    //   ],
    // });

    this.lineBind = device.createBindGroup({
      layout: this.linePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.viewBuf   } },
        { binding: 1, resource: { buffer: this.splineBuf } },
        { binding: 2, resource: { buffer: this.metaBuf   } },
      ],
    });

    // Bake the initial spline (pipelines and bind groups are ready now).
    this.bakeSpline();
  }

  // Dispatch the compute shader to evaluate the Catmull-Rom spline into
  // splineBuf. Called once at init and again whenever data changes.
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
  }

  private attachEvents(): void {
    const sync = () => {
      this.cssWidth  = this.canvas.clientWidth  || 1;
      this.cssHeight = this.canvas.clientHeight || 1;
      if (this.viewInitialized) {
        const dpr = this.opts.pixelRatio;
        const w   = Math.max(1, Math.floor(this.cssWidth  * dpr));
        const h   = Math.max(1, Math.floor(this.cssHeight * dpr));
        this.writeViewViewportStep(w, h);
      }
      this.requestRender();
    };
    sync();
    new ResizeObserver(sync).observe(this.canvas);

    this.canvas.addEventListener('mousedown', (e) => {
      this.dragging   = true;
      this.lastMouseX = e.clientX;
      this.lastMouseY = e.clientY;
      this.canvas.style.cursor = 'grabbing';
    });
    window.addEventListener('mouseup', () => {
      this.dragging = false;
      this.canvas.style.cursor = '';
    });
    window.addEventListener('mousemove', (e) => {
      if (!this.dragging) return;
      const dx = e.clientX - this.lastMouseX;
      const dy = e.clientY - this.lastMouseY;
      this.lastMouseX = e.clientX;
      this.lastMouseY = e.clientY;
      this.panByCssPixels(dx, dy);
    });
    this.canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      const rect   = this.canvas.getBoundingClientRect();
      const factor = Math.exp(-e.deltaY * 0.0015);
      this.zoomAt(e.clientX - rect.left, e.clientY - rect.top, factor);
    }, { passive: false });
    this.canvas.addEventListener('dblclick', () => this.resetView());
  }

  // ---- view math -----------------------------------------------------------

  private panByCssPixels(dxCss: number, dyCss: number): void {
    const dx = (dxCss / this.cssWidth)  * (this.dataMaxX - this.dataMinX);
    const dy = (dyCss / this.cssHeight) * (this.dataMaxY - this.dataMinY);
    this.dataMinX -= dx; this.dataMaxX -= dx;
    this.dataMinY += dy; this.dataMaxY += dy;
    this.writeViewOffset();
    this.requestRender();
  }

  private zoomAt(cssX: number, cssY: number, factor: number): void {
    const fx = cssX / this.cssWidth;
    const fy = 1 - cssY / this.cssHeight;
    const ax = this.dataMinX + fx * (this.dataMaxX - this.dataMinX);
    const ay = this.dataMinY + fy * (this.dataMaxY - this.dataMinY);
    this.dataMinX = ax + (this.dataMinX - ax) / factor;
    this.dataMaxX = ax + (this.dataMaxX - ax) / factor;
    this.dataMinY = ay + (this.dataMinY - ay) / factor;
    this.dataMaxY = ay + (this.dataMaxY - ay) / factor;
    this.writeViewScaleOffsetStep();
    this.requestRender();
  }

  // ---- uniform helpers -----------------------------------------------------
  // The staging ArrayBuffer is a CPU mirror of viewBuf, kept current by every
  // write helper so that partial writes can read unchanged bytes from staging.

  private viewStaging  = new ArrayBuffer(VIEW_BYTES);
  private viewStagingF = new Float32Array(this.viewStaging);
  private viewStagingU = new Uint32Array(this.viewStaging);

  // Drag: only the offset (data-domain centre) changes.
  private writeViewOffset(): void {
    const f = this.viewStagingF;
    f[2] = (this.dataMinX + this.dataMaxX) * 0.5;
    f[3] = (this.dataMinY + this.dataMaxY) * 0.5;
    this.device.queue.writeBuffer(this.viewBuf, 8, this.viewStaging, 8, 8);
  }

  // Zoom: scale, offset, and gridStep change; viewport stays.
  private writeViewScaleOffsetStep(): void {
    const f = this.viewStagingF;
    f[0] = 2 / (this.dataMaxX - this.dataMinX);
    f[1] = 2 / (this.dataMaxY - this.dataMinY);
    f[2] = (this.dataMinX + this.dataMaxX) * 0.5;
    f[3] = (this.dataMinY + this.dataMaxY) * 0.5;
    // f[4], f[5] (viewport) remain current in staging from last writeViewAll/writeViewViewportStep
    f[6] = niceStep(this.dataMaxX - this.dataMinX, this.cssWidth,  100);
    f[7] = niceStep(this.dataMaxY - this.dataMinY, this.cssHeight, 80);
    this.device.queue.writeBuffer(this.viewBuf, 0, this.viewStaging, 0, 32);
  }

  // Resize: viewport and gridStep change; scale and offset stay.
  private writeViewViewportStep(w: number, h: number): void {
    const f = this.viewStagingF;
    f[4] = w; f[5] = h;
    f[6] = niceStep(this.dataMaxX - this.dataMinX, this.cssWidth,  100);
    f[7] = niceStep(this.dataMaxY - this.dataMinY, this.cssHeight, 80);
    this.device.queue.writeBuffer(this.viewBuf, 16, this.viewStaging, 16, 16);
  }

  // Init / reset / setData: write the full 40-byte struct.
  private writeViewAll(w: number, h: number): void {
    const f = this.viewStagingF;
    const u = this.viewStagingU;
    f[0] = 2 / (this.dataMaxX - this.dataMinX);
    f[1] = 2 / (this.dataMaxY - this.dataMinY);
    f[2] = (this.dataMinX + this.dataMaxX) * 0.5;
    f[3] = (this.dataMinY + this.dataMaxY) * 0.5;
    f[4] = w; f[5] = h;
    f[6] = niceStep(this.dataMaxX - this.dataMinX, this.cssWidth,  100);
    f[7] = niceStep(this.dataMaxY - this.dataMinY, this.cssHeight, 80);
    u[8] = this.series.pointCount;
    u[9] = this.series.seriesCount;
    this.device.queue.writeBuffer(this.viewBuf, 0, this.viewStaging);
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

    // pass.setPipeline(this.gridPipeline);
    // pass.setBindGroup(0, this.gridBind);
    // pass.draw(3);

    pass.setPipeline(this.linePipeline);
    pass.setBindGroup(0, this.lineBind);
    if (this.series.pointCount >= 2) {
      pass.draw(2 * (SUBDIVS * (this.series.pointCount - 1) + 1), this.series.seriesCount);
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
