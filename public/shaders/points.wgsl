// Point renderer — draws per-series scatter points.
// Draw call: draw(6, pointCount * seriesCount)
// instance_index = seriesIdx * pointCount + pointIdx

struct View {
  scale:       vec2f,
  offset:      vec2f,
  viewport:    vec2f,
  gridStep:    vec2f,
  pointCount:  u32,
  seriesCount: u32,
};

struct SeriesMeta {
  color:      vec4f,
  styling:    vec4f,   // x=lineWidth px, y=axisIndex(bitcast u32), z=pointSize px, w=pointShape(0=none,1=circle,2=triangle,3=square)
  pointColor: vec4f,
};

struct AxisData {
  yScale:  f32,
  yOffset: f32,
  _pad0:   f32,
  _pad1:   f32,
  color:   vec4f,
};

@group(0) @binding(0) var<uniform>      view:        View;
@group(0) @binding(1) var<storage,read> x:           array<f32>;
@group(0) @binding(2) var<storage,read> y:           array<f32>;
@group(0) @binding(3) var<storage,read> series_meta: array<SeriesMeta>;
@group(0) @binding(4) var<storage,read> axes:        array<AxisData>;

struct VertexOut {
  @builtin(position) pos:   vec4f,
  @location(0)       uv:    vec2f,
  @location(1)       color: vec4f,
  @location(2)       shape: f32,  // 1=circle, 2=triangle, 3=square
};

const QUAD = array<vec2f, 6>(
  vec2f(-1, -1), vec2f( 1, -1), vec2f(-1,  1),
  vec2f(-1,  1), vec2f( 1, -1), vec2f( 1,  1),
);

@vertex fn vs(
  @builtin(vertex_index)   vi: u32,
  @builtin(instance_index) ii: u32,
) -> VertexOut {
  let pointIdx  = ii % view.pointCount;
  let seriesIdx = ii / view.pointCount;

  let m     = series_meta[seriesIdx];
  let shape = m.styling.w;
  let size  = m.styling.z;
  let aIdx  = bitcast<u32>(m.styling.y);
  let ax    = axes[aIdx];

  var out: VertexOut;
  out.uv    = QUAD[vi];
  out.color = m.pointColor;
  out.shape = shape;

  let dataX = x[pointIdx];
  let dataY = y[seriesIdx * view.pointCount + pointIdx];

  let ndcX = (dataX - view.offset.x) * view.scale.x;
  let ndcY = (dataY - ax.yOffset)    * ax.yScale;

  let pxToNdc = 2.0 / view.viewport;
  let offset  = QUAD[vi] * (size * 0.5) * pxToNdc;

  out.pos = vec4f(ndcX + offset.x, ndcY + offset.y, 0.0, 1.0);
  return out;
}

@fragment fn fs(in: VertexOut) -> @location(0) vec4f {
  let shape = u32(round(in.shape));
  var alpha = in.color.a;

  if shape == 1u {
    // Circle SDF
    let d = length(in.uv);
    if d > 1.0 { discard; }
    alpha *= 1.0 - smoothstep(0.8, 1.0, d);
  } else if shape == 2u {
    // Equilateral triangle pointing up, circumscribed in unit circle.
    // Vertices: (0,1), (-√3/2,-½), (√3/2,-½).
    // Edge half-plane distances (positive = inside):
    let k  = 1.7320508; // sqrt(3)
    let d1 = in.uv.y + 0.5;
    let d2 = 1.0 - k * in.uv.x - in.uv.y;
    let d3 = 1.0 + k * in.uv.x - in.uv.y;
    let d  = min(d1, min(d2, d3));
    if d < -0.06 { discard; }
    alpha *= smoothstep(-0.06, 0.06, d);
  } else if shape == 3u {
    // Square with AA at edges
    let d = max(abs(in.uv.x), abs(in.uv.y));
    if d > 1.0 { discard; }
    alpha *= 1.0 - smoothstep(0.85, 1.0, d);
  }

  return vec4f(in.color.rgb, alpha);
}
