// Thick-line renderer. Each draw instance corresponds to one segment of one
// series (instance count = seriesCount * (pointCount - 1)). The vertex shader
// reads the two endpoints from the X / Y storage buffers, transforms them into
// clip space via the view uniform, and extrudes a 6-vertex quad perpendicular
// to the segment in *pixel* space — so line width is consistent regardless of
// zoom.

struct View {
  // Data-domain → NDC: ndc = (data - offset) * scale.  scale.y is negated
  // already so that positive Y is up.
  scale:        vec2f,
  offset:       vec2f,
  // Viewport in physical pixels — needed to convert pixel widths to NDC.
  viewport:     vec2f,
  pointCount:   u32,
  seriesCount:  u32,
};

struct SeriesMeta {
  color:   vec4f,
  styling: vec4f, // x = lineWidth (px), y/z/w reserved
};

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<storage, read> xs: array<f32>;
@group(0) @binding(2) var<storage, read> ys: array<f32>;
@group(0) @binding(3) var<storage, read> meta: array<SeriesMeta>;

struct VertexOut {
  @builtin(position) pos: vec4f,
  @location(0) color: vec4f,
  @location(1) edge:  f32,  // -1..1 across the line, for AA
};

// Six-vertex quad. Columns: which endpoint (a or b), which side (-1 / +1).
//   v0: a, -1     v1: b, -1     v2: a, +1
//   v3: a, +1     v4: b, -1     v5: b, +1
const QUAD_END  = array<f32, 6>(0.0, 1.0, 0.0, 0.0, 1.0, 1.0);
const QUAD_SIDE = array<f32, 6>(-1.0, -1.0, 1.0, 1.0, -1.0, 1.0);

@vertex fn vs(
  @builtin(vertex_index)   vi: u32,
  @builtin(instance_index) ii: u32,
) -> VertexOut {
  let segPerSeries = view.pointCount - 1u;
  let series = ii / segPerSeries;
  let seg    = ii % segPerSeries;

  let yBase = series * view.pointCount;
  let pa = vec2f(xs[seg],      ys[yBase + seg]);
  let pb = vec2f(xs[seg + 1u], ys[yBase + seg + 1u]);

  // To NDC.
  let ax = (pa - view.offset) * view.scale;
  let bx = (pb - view.offset) * view.scale;

  // Extrude perpendicular in pixel space.  Convert NDC delta to pixels using
  // half the viewport, perpendicularize, then convert pixel offset back to NDC.
  let ndcToPx = view.viewport * 0.5;
  let pxToNdc = vec2f(2.0) / view.viewport;

  let dirPx = normalize((bx - ax) * ndcToPx + vec2f(1e-6, 0.0));
  let nrmPx = vec2f(-dirPx.y, dirPx.x);

  let m = meta[series];
  let halfW = m.styling.x * 0.5;

  let end  = QUAD_END[vi];
  let side = QUAD_SIDE[vi];

  let basePx  = mix(ax, bx, end) * ndcToPx;
  let offsetPx = nrmPx * (halfW * side);
  let posNdc   = (basePx + offsetPx) * pxToNdc;

  var out: VertexOut;
  out.pos   = vec4f(posNdc, 0.0, 1.0);
  out.color = m.color;
  out.edge  = side;
  return out;
}

@fragment fn fs(in: VertexOut) -> @location(0) vec4f {
  // Edge antialias — fade the outermost ~1px of the strip.
  let aa = 1.0 - smoothstep(0.75, 1.0, abs(in.edge));
  return vec4f(in.color.rgb, in.color.a * aa);
}
