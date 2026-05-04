import { type SeriesBufferView, createSeriesBuffer, setStyle, setAxisConfig, recomputeExtents } from './format.js';

// generateLines — groups[a] = series count for axis a.  Each axis group gets a
// different amplitude scale so independent Y ranges are obvious in the demo.
export function generateLines(groups: number[], pointCount: number): SeriesBufferView {
  const sb = createSeriesBuffer(groups, pointCount);
  const totalSeries = sb.seriesCount;

  for (let i = 0; i < pointCount; i++) {
    sb.x[i] = i / (pointCount - 1);
  }

  let s = 0;
  for (let a = 0; a < groups.length; a++) {
    const groupAmp = Math.pow(10, a * 2); // axis 0: ±1, axis 1: ±100, axis 2: ±10000 …

    for (let g = 0; g < groups[a]!; g++, s++) {
      const phase = (s / totalSeries) * Math.PI * 2;
      const freq  = 1 + s * 0.7;
      const amp   = groupAmp * (0.7 + Math.random() * 0.4);
      const drift = (g - groups[a]! / 2) * groupAmp * 0.4;

      let walk = 0;
      for (let i = 0; i < pointCount; i++) {
        const t = sb.x[i]!;
        walk += (Math.random() - 0.5) * 0.05 * groupAmp;
        sb.y[s * pointCount + i] = drift + amp * Math.sin(phase + freq * t * Math.PI * 2) + walk;
      }

      const hue = s / totalSeries;
      const [r, gr, b] = hslToRgb(hue, 0.7, 0.6);
      setStyle(sb, s, { color: [r, gr, b, 1], width: 2.5 });

      if (g === 0) setAxisConfig(sb, a, { color: [r, gr, b, 1] });
    }
  }

  recomputeExtents(sb);
  return sb;
}

export function generateClusters(
  clusterCount: number,
  pointsPerCluster: number,
  spread = 0.12,
): { positions: Float32Array<ArrayBuffer>; colors: Float32Array<ArrayBuffer>; count: number } {
  const count = clusterCount * pointsPerCluster;
  const positions: Float32Array<ArrayBuffer> = new Float32Array(count * 2);
  const colors: Float32Array<ArrayBuffer>    = new Float32Array(count * 4);

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
