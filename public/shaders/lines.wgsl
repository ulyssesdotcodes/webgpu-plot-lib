// Thick-line renderer — triangle-strip, reads pre-baked Catmull-Rom positions.
//
// Draw call: draw(2 * (SUBDIVS * (pointCount - 1) + 1), seriesCount)

const SUBDIVS: u32 = 4u;

struct View {
  scale:       vec2f,   // ndc = (data - offset) * scale  (scale.y unused; Y handled per-axis)
  offset:      vec2f,
  viewport:    vec2f,
  gridStep:    vec2f,
  pointCount:  u32,
  seriesCount: u32,
};

struct SeriesMeta {
  color:      vec4f,
  styling:    vec4f,   // x=lineWidth px, y=axisIndex(bitcast u32), z=pointSize px, w=pointShape(0..3)
  pointColor: vec4f,
};

// Per-axis Y transform and label color (matches chart.ts axesBuf layout).
struct AxisData {
  yScale:  f32,
  yOffset: f32,
  _pad0:   f32,
  _pad1:   f32,
  color:   vec4f,
};

@group(0) @binding(0) var<uniform>      view:        View;
@group(0) @binding(1) var<storage,read> spline:      array<vec2f>;
@group(0) @binding(2) var<storage,read> series_meta: array<SeriesMeta>;
@group(0) @binding(3) var<storage,read> axes:        array<AxisData>;

struct VertexOut {
  @builtin(position) pos:   vec4f,
  @location(0)       color: vec4f,
  @location(1)       edge:  f32,
};

@vertex fn vs(
  @builtin(vertex_index)   vi: u32,
  @builtin(instance_index) ii: u32,
) -> VertexOut {
  let totalSamples = SUBDIVS * (view.pointCount - 1u) + 1u;

  let pair = vi >> 1u;
  let side = f32(vi & 1u) * 2.0 - 1.0;

  let base = ii * totalSamples;
  let pos  = spline[base + pair];

  let prev = select(pair - 1u, 0u,                pair == 0u);
  let next = select(pair + 1u, totalSamples - 1u, pair == totalSamples - 1u);
  let tanDat = spline[base + next] - spline[base + prev];

  let ndcToPx = view.viewport * 0.5;
  let pxToNdc = vec2f(2.0) / view.viewport;

  let m    = series_meta[ii];
  let aIdx = bitcast<u32>(m.styling.y);
  let ax   = axes[aIdx];

  let posNdc = vec2f(
    (pos.x - view.offset.x) * view.scale.x,
    (pos.y - ax.yOffset)    * ax.yScale,
  );

  let scaleXY = vec2f(view.scale.x, ax.yScale);
  let tanPx   = normalize(tanDat * scaleXY * ndcToPx + vec2f(1e-6, 1e-6));
  let nrmPx   = vec2f(-tanPx.y, tanPx.x);

  let halfW  = m.styling.x * 0.5;
  let posOut = posNdc + nrmPx * (halfW * side) * pxToNdc;

  var out: VertexOut;
  out.pos   = vec4f(posOut, 0.0, 1.0);
  out.color = m.color;
  out.edge  = side;
  return out;
}

@fragment fn fs(in: VertexOut) -> @location(0) vec4f {
  let aa = 1.0 - smoothstep(0.75, 1.0, abs(in.edge));
  return vec4f(in.color.rgb, in.color.a * aa);
}
