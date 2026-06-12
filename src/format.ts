// SeriesBuffer — a self-describing, single-ArrayBuffer wire format for line-chart
// data. One buffer carries the header, a shared X column, an N×M Y matrix,
// per-series styling metadata, and per-axis config. Because everything lives in
// one ArrayBuffer the whole payload is `postMessage`-transferable.
//
// Layout (all little-endian, f32 / u32):
//
//   [ Header 64B ][ X pointCount*4 ][ Y seriesCount*pointCount*4 ][ Meta seriesCount*32 ][ Axes axisCount*32 ]
//
// Header (16 × u32/f32 = 64 bytes):
//   0   magic        u32   'VDV2' (0x32564456)
//   1   version      u32   2
//   2   flags        u32   bit 0 = sharedX (currently always set)
//   3   seriesCount  u32
//   4   pointCount   u32   per series
//   5   xOffset      u32   byte offset of X array within buffer
//   6   yOffset      u32   byte offset of Y matrix within buffer
//   7   metaOffset   u32   byte offset of per-series meta
//   8   xMin         f32
//   9   xMax         f32
//   10  yMin         f32   global union across all axes
//   11  yMax         f32
//   12  axisCount    u32
//   13  axisOffset   u32   byte offset of per-axis config
//   14-15  reserved
//
// Y matrix is series-major: y[s, i] lives at yOffset + (s*pointCount + i) * 4.
// Meta is an array of 48B per series (3 × vec4):
//   floats 0-3: line color RGBA
//   float  4:   lineWidth (px)
//   float  5:   axisIndex (effectively u32)
//   float  6:   pointSize (px); 0 = no points
//   float  7:   pointShape (0=none, 1=circle, 2=triangle, 3=square)
//   floats 8-11: point color RGBA
// Axes is an array of 32B per axis:
//   floats 0-3: label tint color RGBA
//   floats 4-7: padding

export const MAGIC       = 0x32564456; // 'VDV2' little-endian
export const VERSION     = 2;
export const HEADER_BYTES = 64;
export const META_STRIDE  = 48; // bytes per series (12 floats)
export const AXIS_STRIDE  = 32; // bytes per axis

export const FLAG_SHARED_X = 1 << 0;

export type PointShape = 'circle' | 'triangle' | 'square';

export interface PointStyle {
  size?:  number;
  shape?: PointShape;
  color?: [number, number, number, number];
}

export interface SeriesStyle {
  color:      [number, number, number, number];
  width:      number;
  axisIndex?: number;
  points?:    PointStyle;
}

export interface AxisConfig {
  color: [number, number, number, number];
}

export interface SeriesBufferView {
  buffer:   ArrayBuffer;
  header:   Uint32Array<ArrayBuffer>;
  headerF:  Float32Array<ArrayBuffer>;
  x:        Float32Array<ArrayBuffer>;
  y:        Float32Array<ArrayBuffer>;
  meta:     Float32Array<ArrayBuffer>;
  axes:     Float32Array<ArrayBuffer>; // axisCount * 8 floats
  seriesCount: number;
  pointCount:  number;
  axisCount:   number;
}

// groups[a] = number of series on axis a.  Total series = sum(groups).
export function createSeriesBuffer(groups: number[], pointCount: number): SeriesBufferView {
  if (groups.length === 0 || groups.some(g => g <= 0))
    throw new Error('each group must have at least 1 series');
  if (pointCount <= 0) throw new Error('pointCount must be > 0');

  const seriesCount = groups.reduce((a, b) => a + b, 0);
  const axisCount   = groups.length;
  const xBytes      = pointCount  * 4;
  const yBytes      = seriesCount * pointCount * 4;
  const metaBytes   = seriesCount * META_STRIDE;
  const axesBytes   = axisCount   * AXIS_STRIDE;
  const totalBytes  = HEADER_BYTES + xBytes + yBytes + metaBytes + axesBytes;

  const buffer  = new ArrayBuffer(totalBytes);
  const header  = new Uint32Array(buffer, 0, 16);
  const headerF = new Float32Array(buffer, 0, 16);

  header[0]  = MAGIC;
  header[1]  = VERSION;
  header[2]  = FLAG_SHARED_X;
  header[3]  = seriesCount;
  header[4]  = pointCount;
  header[5]  = HEADER_BYTES;
  header[6]  = HEADER_BYTES + xBytes;
  header[7]  = HEADER_BYTES + xBytes + yBytes;
  header[12] = axisCount;
  header[13] = HEADER_BYTES + xBytes + yBytes + metaBytes;

  const x    = new Float32Array(buffer, header[5]!,  pointCount);
  const y    = new Float32Array(buffer, header[6]!,  seriesCount * pointCount);
  const meta = new Float32Array(buffer, header[7]!,  seriesCount * 12);
  const axes = new Float32Array(buffer, header[13]!, axisCount   * 8);

  // Default series meta: white, 2.5 px, no points, axis assigned from groups.
  let s = 0;
  for (let a = 0; a < axisCount; a++) {
    for (let g = 0; g < groups[a]!; g++, s++) {
      const o = s * 12;
      meta[o] = 1; meta[o+1] = 1; meta[o+2] = 1; meta[o+3] = 1; // line color: white
      meta[o+4] = 2.5;  // lineWidth
      meta[o+5] = a;    // axisIndex
      meta[o+6] = 0;    // pointSize: 0 = no points
      meta[o+7] = 0;    // pointShape: none
      meta[o+8] = 1; meta[o+9] = 1; meta[o+10] = 1; meta[o+11] = 1; // pointColor: white
    }
  }

  // Default axis config: white labels.
  for (let a = 0; a < axisCount; a++) {
    const o = a * 8;
    axes[o] = 1; axes[o+1] = 1; axes[o+2] = 1; axes[o+3] = 1;
  }

  return { buffer, header, headerF, x, y, meta, axes, seriesCount, pointCount, axisCount };
}

export function viewSeriesBuffer(buffer: ArrayBuffer): SeriesBufferView {
  const header = new Uint32Array(buffer, 0, 16);
  if (header[0] !== MAGIC)   throw new Error('SeriesBuffer: bad magic (expected VDV2)');
  if (header[1] !== VERSION) throw new Error(`SeriesBuffer: unsupported version ${header[1]!}`);

  const headerF    = new Float32Array(buffer, 0, 16);
  const seriesCount = header[3]!;
  const pointCount  = header[4]!;
  const axisCount   = header[12]!;

  return {
    buffer, header, headerF,
    x:    new Float32Array(buffer, header[5]!,  pointCount),
    y:    new Float32Array(buffer, header[6]!,  seriesCount * pointCount),
    meta: new Float32Array(buffer, header[7]!,  seriesCount * 12),
    axes: new Float32Array(buffer, header[13]!, axisCount   * 8),
    seriesCount, pointCount, axisCount,
  };
}

const POINT_SHAPE_MAP: Record<PointShape, number> = { circle: 1, triangle: 2, square: 3 };

export function setStyle(view: SeriesBufferView, series: number, style: SeriesStyle): void {
  const o = series * 12;
  view.meta[o]   = style.color[0];
  view.meta[o+1] = style.color[1];
  view.meta[o+2] = style.color[2];
  view.meta[o+3] = style.color[3];
  view.meta[o+4] = style.width;
  if (style.axisIndex !== undefined) view.meta[o+5] = style.axisIndex;
  if (style.points !== undefined) {
    view.meta[o+6] = style.points.size  ?? 6;
    view.meta[o+7] = style.points.shape ? POINT_SHAPE_MAP[style.points.shape] : 1;
    const pc = style.points.color ?? style.color;
    view.meta[o+8]  = pc[0];
    view.meta[o+9]  = pc[1];
    view.meta[o+10] = pc[2];
    view.meta[o+11] = pc[3];
  }
}

export function setAxisConfig(view: SeriesBufferView, axisIndex: number, config: AxisConfig): void {
  const o = axisIndex * 8;
  view.axes[o]   = config.color[0];
  view.axes[o+1] = config.color[1];
  view.axes[o+2] = config.color[2];
  view.axes[o+3] = config.color[3];
}

export function getSeriesAxis(view: SeriesBufferView, series: number): number {
  return Math.round(view.meta[series * 12 + 5]!);
}

export function getAxisColor(view: SeriesBufferView, axisIndex: number): [number, number, number, number] {
  const o = axisIndex * 8;
  return [view.axes[o]!, view.axes[o+1]!, view.axes[o+2]!, view.axes[o+3]!];
}

export function recomputeExtents(view: SeriesBufferView): void {
  let xMin = Infinity, xMax = -Infinity;
  for (let i = 0; i < view.pointCount; i++) {
    const v = view.x[i]!;
    if (v < xMin) xMin = v;
    if (v > xMax) xMax = v;
  }
  let yMin = Infinity, yMax = -Infinity;
  for (let i = 0; i < view.y.length; i++) {
    const v = view.y[i]!;
    if (v < yMin) yMin = v;
    if (v > yMax) yMax = v;
  }
  view.headerF[8]  = xMin;
  view.headerF[9]  = xMax;
  view.headerF[10] = yMin;
  view.headerF[11] = yMax;
}

export function getExtents(view: SeriesBufferView): { xMin: number; xMax: number; yMin: number; yMax: number } {
  return {
    xMin: view.headerF[8]!,
    xMax: view.headerF[9]!,
    yMin: view.headerF[10]!,
    yMax: view.headerF[11]!,
  };
}

// CPU-side per-axis Y extents for the initial view; dynamic extents use the GPU.
export function getAxisCpuExtents(view: SeriesBufferView): Array<{ yMin: number; yMax: number }> {
  const result: Array<{ yMin: number; yMax: number }> = [];
  for (let a = 0; a < view.axisCount; a++) {
    let yMin = Infinity, yMax = -Infinity;
    for (let s = 0; s < view.seriesCount; s++) {
      if (getSeriesAxis(view, s) !== a) continue;
      const base = s * view.pointCount;
      for (let i = 0; i < view.pointCount; i++) {
        const v = view.y[base + i]!;
        if (v < yMin) yMin = v;
        if (v > yMax) yMax = v;
      }
    }
    result.push({ yMin, yMax });
  }
  return result;
}
