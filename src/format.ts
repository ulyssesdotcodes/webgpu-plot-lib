// SeriesBuffer — a self-describing, single-ArrayBuffer wire format for line-chart
// data. One buffer carries the header, a shared X column, an N×M Y matrix, and
// per-series styling metadata. Because everything lives in one ArrayBuffer the
// whole payload is `postMessage`-transferable to a worker for decimation /
// extent computation, and each region is uploaded directly to the GPU as a
// storage buffer (no per-frame copies).
//
// Layout (all little-endian, f32 / u32):
//
//   [ Header  64B ][ X  pointCount*4 ][ Y seriesCount*pointCount*4 ][ Meta seriesCount*32 ]
//
// Header (16 × u32/f32 = 64 bytes):
//   0   magic        u32   'VDV1' (0x31564456)
//   1   version      u32
//   2   flags        u32   bit 0 = sharedX (currently always set)
//   3   seriesCount  u32
//   4   pointCount   u32   per series
//   5   xOffset      u32   byte offset of X array within buffer
//   6   yOffset      u32   byte offset of Y matrix within buffer
//   7   metaOffset   u32   byte offset of per-series meta
//   8   xMin         f32
//   9   xMax         f32
//   10  yMin         f32
//   11  yMax         f32
//   12-15  reserved
//
// Y matrix is series-major: y[s, i] lives at yOffset + (s*pointCount + i) * 4.
// Meta is an array of vec4+vec4 (32B per series) — { color.rgba, [width, _, _, _] }
// — aligned for direct binding as a uniform/storage buffer.

export const MAGIC = 0x31564456; // 'VDV1' little-endian
export const VERSION = 1;
export const HEADER_BYTES = 64;
export const META_STRIDE = 32; // bytes per series

export const FLAG_SHARED_X = 1 << 0;

export interface SeriesStyle {
  color: [number, number, number, number]; // rgba 0..1
  width: number;                            // device-independent pixels
}

export interface SeriesBufferView {
  buffer: ArrayBuffer;
  header: Uint32Array;     // u32 view of the header (length 16)
  headerF: Float32Array;   // f32 view of the header (length 16)
  x: Float32Array;         // pointCount
  y: Float32Array;         // seriesCount * pointCount, series-major
  meta: Float32Array;      // seriesCount * 8 floats
  seriesCount: number;
  pointCount: number;
}

export function createSeriesBuffer(seriesCount: number, pointCount: number): SeriesBufferView {
  if (seriesCount <= 0 || pointCount <= 0) {
    throw new Error('seriesCount and pointCount must be > 0');
  }

  const xBytes = pointCount * 4;
  const yBytes = seriesCount * pointCount * 4;
  const metaBytes = seriesCount * META_STRIDE;
  const totalBytes = HEADER_BYTES + xBytes + yBytes + metaBytes;

  const buffer = new ArrayBuffer(totalBytes);
  const header = new Uint32Array(buffer, 0, 16);
  const headerF = new Float32Array(buffer, 0, 16);

  header[0] = MAGIC;
  header[1] = VERSION;
  header[2] = FLAG_SHARED_X;
  header[3] = seriesCount;
  header[4] = pointCount;
  header[5] = HEADER_BYTES;
  header[6] = HEADER_BYTES + xBytes;
  header[7] = HEADER_BYTES + xBytes + yBytes;

  const x = new Float32Array(buffer, HEADER_BYTES, pointCount);
  const y = new Float32Array(buffer, HEADER_BYTES + xBytes, seriesCount * pointCount);
  const meta = new Float32Array(buffer, HEADER_BYTES + xBytes + yBytes, seriesCount * 8);

  // Default style — opaque white, 1.5px lines.
  for (let s = 0; s < seriesCount; s++) {
    const o = s * 8;
    meta[o] = 1; meta[o + 1] = 1; meta[o + 2] = 1; meta[o + 3] = 1;
    meta[o + 4] = 1.5;
  }

  return { buffer, header, headerF, x, y, meta, seriesCount, pointCount };
}

export function viewSeriesBuffer(buffer: ArrayBuffer): SeriesBufferView {
  const header = new Uint32Array(buffer, 0, 16);
  if (header[0] !== MAGIC) throw new Error('SeriesBuffer: bad magic');
  if (header[1] !== VERSION) throw new Error(`SeriesBuffer: unsupported version ${header[1]}`);

  const headerF = new Float32Array(buffer, 0, 16);
  const seriesCount = header[3]!;
  const pointCount = header[4]!;
  const xOffset = header[5]!;
  const yOffset = header[6]!;
  const metaOffset = header[7]!;

  return {
    buffer,
    header,
    headerF,
    x: new Float32Array(buffer, xOffset, pointCount),
    y: new Float32Array(buffer, yOffset, seriesCount * pointCount),
    meta: new Float32Array(buffer, metaOffset, seriesCount * 8),
    seriesCount,
    pointCount,
  };
}

export function setStyle(view: SeriesBufferView, series: number, style: SeriesStyle): void {
  const o = series * 8;
  view.meta[o]     = style.color[0];
  view.meta[o + 1] = style.color[1];
  view.meta[o + 2] = style.color[2];
  view.meta[o + 3] = style.color[3];
  view.meta[o + 4] = style.width;
}

// Compute and store extents in the header. Cheap on the main thread; can be
// moved to a worker for very large buffers — the buffer is transferable.
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
