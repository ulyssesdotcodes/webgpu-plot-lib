// Thick-line renderer — triangle-strip, reads pre-baked Catmull-Rom positions.
//
// The spline evaluation runs once per data change in spline.wgsl; this shader
// only reads pre-baked positions, applies the view transform, and extrudes
// each vertex perpendicular to the local tangent.
//
// Draw call: draw(2 * (SUBDIVS * (pointCount - 1) + 1), seriesCount)
//   Each pair of vertices (vi = 2*pair, 2*pair+1) is one left/right edge.
//   Triangle-strip wires adjacent pairs into quads automatically.

const SUBDIVS: u32 = 4u;  // must match spline.wgsl and chart.ts

// ---- uniforms ---------------------------------------------------------------

// Matches the 40-byte View layout in chart.ts.
struct View {
  scale:       vec2f,   // ndc = (data - offset) * scale
  offset:      vec2f,   // data-domain centre of the visible window
  viewport:    vec2f,   // canvas size in physical pixels
  gridStep:    vec2f,   // (grid shader only; kept so shaders share one struct)
  pointCount:  u32,
  seriesCount: u32,
};

// 32 bytes per series: rgba color + vec4 styling (x = lineWidth in px).
struct SeriesMeta {
  color:   vec4f,
  styling: vec4f,
};

@group(0) @binding(0) var<uniform>      view:        View;
@group(0) @binding(1) var<storage,read> spline:      array<vec2f>;  // pre-baked positions, data space
@group(0) @binding(2) var<storage,read> series_meta: array<SeriesMeta>;

// ---- interpolants -----------------------------------------------------------

struct VertexOut {
  @builtin(position) pos:   vec4f,
  @location(0)       color: vec4f,
  @location(1)       edge:  f32,  // -1 (left) or +1 (right), for AA
};

// ---- vertex shader ----------------------------------------------------------

@vertex fn vs(
  @builtin(vertex_index)   vi: u32,  // 0 .. 2*(SUBDIVS*(N-1)+1) - 1
  @builtin(instance_index) ii: u32,  // series index
) -> VertexOut {
  // totalSamples = number of pre-baked positions per series.
  let totalSamples = SUBDIVS * (view.pointCount - 1u) + 1u;

  let pair = vi >> 1u;                   // which sample along the strip
  let side = f32(vi & 1u) * 2.0 - 1.0;  // -1 (left) or +1 (right)

  // Base index into the spline buffer for this series.
  let base = ii * totalSamples;
  let pos  = spline[base + pair];

  // Tangent via centred difference of pre-baked neighbours.
  // The spline is already C1-smooth at every knot, so this gives a good
  // approximation without any cubic math in the vertex shader.
  let prev = select(pair - 1u, 0u,                pair == 0u);
  let next = select(pair + 1u, totalSamples - 1u, pair == totalSamples - 1u);
  let tanDat = spline[base + next] - spline[base + prev];

  // Coordinate conversions.
  let ndcToPx = view.viewport * 0.5;         // NDC delta → pixel delta
  let pxToNdc = vec2f(2.0) / view.viewport;  // pixel delta → NDC delta

  let posNdc = (pos - view.offset) * view.scale;

  // Tangent in pixel space: apply view scale then ndcToPx.
  // Small x-epsilon prevents degenerate zero vectors for vertical segments.
  let tanPx = normalize(tanDat * view.scale * ndcToPx + vec2f(1e-6, 0.0));

  // Left-perpendicular gives the extrusion direction.
  let nrmPx = vec2f(-tanPx.y, tanPx.x);

  let m      = series_meta[ii];
  let halfW  = m.styling.x * 0.5;

  // Extrude in pixel space, convert back to NDC.
  let posOut = posNdc + nrmPx * (halfW * side) * pxToNdc;

  var out: VertexOut;
  out.pos   = vec4f(posOut, 0.0, 1.0);
  out.color = m.color;
  out.edge  = side;
  return out;
}

// ---- fragment shader --------------------------------------------------------

@fragment fn fs(in: VertexOut) -> @location(0) vec4f {
  // Soft anti-aliased edge: full opacity in the inner 75%, fade over ~1 px.
  let aa = 1.0 - smoothstep(0.75, 1.0, abs(in.edge));
  return vec4f(in.color.rgb, in.color.a * aa);
}
