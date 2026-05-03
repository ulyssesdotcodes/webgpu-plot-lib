// Instanced text renderer — one quad (6 vertices) per character.
// CPU pre-computes NDC positions; this shader just places and samples the atlas.

struct CharInst {
  ndcPos:  vec2f,  // NDC top-left corner  (y+ is up)
  ndcSize: vec2f,  // NDC width/height     (both positive)
  uvMin:   vec2f,  // atlas UV of slot top-left
  uvMax:   vec2f,  // atlas UV of slot bottom-right
};

@group(0) @binding(0) var<storage,read> chars: array<CharInst>;
@group(0) @binding(1) var atlas: texture_2d<f32>;
@group(0) @binding(2) var samp:  sampler;

struct VertOut {
  @builtin(position) pos: vec4f,
  @location(0)       uv:  vec2f,
};

// Two CCW triangles covering the unit square [0,1]².
const QUAD = array<vec2f, 6>(
  vec2f(0, 0), vec2f(1, 0), vec2f(0, 1),
  vec2f(1, 0), vec2f(1, 1), vec2f(0, 1),
);

@vertex fn vs(
  @builtin(vertex_index)   vi: u32,
  @builtin(instance_index) ii: u32,
) -> VertOut {
  let q  = QUAD[vi];
  let ch = chars[ii];
  // NDC: x+ right, y+ up.  q.y increases downward in screen space → subtract.
  let ndc = vec2f(
    ch.ndcPos.x + q.x * ch.ndcSize.x,
    ch.ndcPos.y - q.y * ch.ndcSize.y,
  );
  var out: VertOut;
  out.pos = vec4f(ndc, 0.0, 1.0);
  out.uv  = mix(ch.uvMin, ch.uvMax, q);
  return out;
}

@fragment fn fs(in: VertOut) -> @location(0) vec4f {
  let a = textureSample(atlas, samp, in.uv).a;
  return vec4f(1.0, 1.0, 1.0, a);
}
