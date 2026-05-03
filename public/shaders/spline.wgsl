// Compute shader — pre-evaluates Catmull-Rom spline positions into a flat
// buffer of vec2f values (data-space x,y).  Runs once per data change; the
// lines vertex shader reads from this buffer with no cubic math per frame.
//
// Output layout: spline[series * totalSamples + sample] = vec2f(x, y)
// Dispatch:      ceil((seriesCount * totalSamples) / 64) workgroups of 64.

const SUBDIVS: u32 = 4u;  // must match lines.wgsl and chart.ts

struct SplineParams {
  pointCount:   u32,
  seriesCount:  u32,
  totalSamples: u32,  // = SUBDIVS * (pointCount - 1) + 1
  _pad:         u32,
};

@group(0) @binding(0) var<uniform>            params: SplineParams;
@group(0) @binding(1) var<storage,read>       xs:     array<f32>;
@group(0) @binding(2) var<storage,read>       ys:     array<f32>;
@group(0) @binding(3) var<storage,read_write> spline: array<vec2f>;

// Catmull-Rom position — same formula as lines.wgsl had, now run once here.
fn crPos(p0: vec2f, p1: vec2f, p2: vec2f, p3: vec2f, t: f32) -> vec2f {
  let t2 = t * t;
  let t3 = t2 * t;
  return 0.5 * (
    ( 2.0 * p1                       ) +
    (-p0         + p2                ) * t  +
    ( 2.0*p0 - 5.0*p1 + 4.0*p2 - p3 ) * t2 +
    (-p0 + 3.0*p1 - 3.0*p2 + p3     ) * t3
  );
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let totalWork = params.seriesCount * params.totalSamples;
  if gid.x >= totalWork { return; }

  let series = gid.x / params.totalSamples;
  let pair   = gid.x % params.totalSamples;

  // Map (pair) → (segment, t) — same logic as the old vertex shader.
  let N   = params.pointCount;
  var seg = pair / SUBDIVS;
  var t   = f32(pair % SUBDIVS) / f32(SUBDIVS);
  if seg >= N - 1u { seg = N - 2u; t = 1.0; }

  // Four control points, with phantom points clamped at boundaries.
  let i0 = select(seg - 1u, 0u,     seg == 0u);
  let i1 = seg;
  let i2 = seg + 1u;
  let i3 = select(seg + 2u, N - 1u, seg + 2u >= N);

  let yBase = series * N;
  let p0 = vec2f(xs[i0], ys[yBase + i0]);
  let p1 = vec2f(xs[i1], ys[yBase + i1]);
  let p2 = vec2f(xs[i2], ys[yBase + i2]);
  let p3 = vec2f(xs[i3], ys[yBase + i3]);

  spline[series * params.totalSamples + pair] = crPos(p0, p1, p2, p3, t);
}
