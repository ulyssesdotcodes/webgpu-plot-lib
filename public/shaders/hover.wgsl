// Nearest-point picker. One thread per data point; each computes the screen-
// space distance from its point to the cursor and races to write the smallest
// (distance, index) pair into a single u32 via atomicMin.
//
// Packed layout (32 bits):
//   bits 31..20  pixel distance, clamped to [0, 4095]   (upper, sort key)
//   bits 19..0   linear index = series * pointCount + point   (lower, payload)
//
// Caller initialises `result` to 0xFFFFFFFF before each dispatch and decodes
// after the readback maps. dist == 4095 means "no point within ~4k px" — the
// caller usually treats anything past ~30 px as a miss.

struct View {
  scale:        vec2f,
  offset:       vec2f,
  viewport:     vec2f,
  pointCount:   u32,
  seriesCount:  u32,
};

struct Hover {
  mousePx: vec2f,      // physical pixels from canvas top-left
  _pad:    vec2f,
};

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<storage, read> xs: array<f32>;
@group(0) @binding(2) var<storage, read> ys: array<f32>;
@group(0) @binding(3) var<uniform> hover: Hover;
@group(0) @binding(4) var<storage, read_write> result: atomic<u32>;

const DIST_MAX:   f32 = 4095.0;
const INDEX_BITS: u32 = 20u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let total = view.pointCount * view.seriesCount;
  let i = gid.x;
  if i >= total { return; }

  let series = i / view.pointCount;
  let pt     = i % view.pointCount;
  let data = vec2f(xs[pt], ys[series * view.pointCount + pt]);
  let ndc  = (data - view.offset) * view.scale;

  // NDC → screen pixels.  Y axis is flipped (screen Y goes down).
  let screenPx = vec2f(
    (ndc.x * 0.5 + 0.5) * view.viewport.x,
    (1.0 - (ndc.y * 0.5 + 0.5)) * view.viewport.y,
  );

  let dist  = distance(screenPx, hover.mousePx);
  let distQ = u32(min(dist, DIST_MAX));
  let packed = (distQ << INDEX_BITS) | i;
  atomicMin(&result, packed);
}
