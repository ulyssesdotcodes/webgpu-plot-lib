import { type SeriesBufferView, createSeriesBuffer, setStyle, recomputeExtents } from './format.js';

// Small library of generators that produce a fully-populated SeriesBuffer for
// the demo / tests. They write directly into the typed-array views the format
// hands out — no intermediate copies — so the same buffer can be transferred
// to a worker afterward without restructuring.

export function generateLines(seriesCount: number, pointCount: number): SeriesBufferView {
  const sb = createSeriesBuffer(seriesCount, pointCount);

  // X axis: dense uniform grid 0..1 (thousands to millions of points are fine).
  for (let i = 0; i < pointCount; i++) {
    sb.x[i] = i / (pointCount - 1);
  }

  for (let s = 0; s < seriesCount; s++) {
    const phase = (s / seriesCount) * Math.PI * 2;
    const freq  = 1 + s * 0.7;
    const amp   = 0.7 + Math.random() * 0.4;
    const drift = (s - seriesCount / 2) * 0.4;

    let walk = 0;
    for (let i = 0; i < pointCount; i++) {
      const t = sb.x[i]!;
      walk += (Math.random() - 0.5) * 0.05;
      sb.y[s * pointCount + i] = drift + amp * Math.sin(phase + freq * t * Math.PI * 2) + walk;
    }

    const hue = s / seriesCount;
    const [r, g, b] = hslToRgb(hue, 0.7, 0.6);
    setStyle(sb, s, { color: [r, g, b, 1], width: 2.5 });
  }

  recomputeExtents(sb);
  return sb;
}

// Kept from the previous scaffold so the original points demo still has data.
export function generateClusters(
  clusterCount: number,
  pointsPerCluster: number,
  spread = 0.12,
): { positions: Float32Array<ArrayBuffer>; colors: Float32Array<ArrayBuffer>; count: number } {
  const count = clusterCount * pointsPerCluster;
  const positions: Float32Array<ArrayBuffer> = new Float32Array(count * 2);
  const colors: Float32Array<ArrayBuffer> = new Float32Array(count * 4);

  for (let c = 0; c < clusterCount; c++) {
    const cx = Math.random() * 1.6 - 0.8;
    const cy = Math.random() * 1.6 - 0.8;
    const hue = c / clusterCount;
    const [r, g, b] = hslToRgb(hue, 0.7, 0.6);

    for (let p = 0; p < pointsPerCluster; p++) {
      const i = c * pointsPerCluster + p;
      const [dx, dy] = boxMuller();
      positions[i * 2]     = cx + dx * spread;
      positions[i * 2 + 1] = cy + dy * spread;
      colors[i * 4]     = r;
      colors[i * 4 + 1] = g;
      colors[i * 4 + 2] = b;
      colors[i * 4 + 3] = 0.85;
    }
  }

  return { positions, colors, count };
}

function boxMuller(): [number, number] {
  const u = Math.random() || 1e-10;
  const v = Math.random();
  const mag = Math.sqrt(-2 * Math.log(u));
  return [mag * Math.cos(2 * Math.PI * v), mag * Math.sin(2 * Math.PI * v)];
}

function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  const a = s * Math.min(l, 1 - l);
  const f = (n: number) => {
    const k = (n + h * 12) % 12;
    return l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1));
  };
  return [f(0), f(8), f(4)];
}
