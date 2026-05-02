@group(0) @binding(0) var<storage, read> positions: array<vec2f>;
@group(0) @binding(1) var<storage, read> colors: array<vec4f>;

struct VertexOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) color: vec4f,
}

const POINT_SIZE: f32 = 0.015;

const QUAD = array<vec2f, 6>(
  vec2f(-1, -1), vec2f( 1, -1), vec2f(-1,  1),
  vec2f(-1,  1), vec2f( 1, -1), vec2f( 1,  1),
);

@vertex fn vs(
  @builtin(vertex_index) vi: u32,
  @builtin(instance_index) ii: u32,
) -> VertexOut {
  let q = QUAD[vi];
  var out: VertexOut;
  out.pos = vec4f(positions[ii] + q * POINT_SIZE, 0.0, 1.0);
  out.uv = q;
  out.color = colors[ii];
  return out;
}

@fragment fn fs(in: VertexOut) -> @location(0) vec4f {
  let d = length(in.uv);
  if d > 1.0 { discard; }
  let alpha = in.color.a * (1.0 - smoothstep(0.6, 1.0, d));
  return vec4f(in.color.rgb, alpha);
}
