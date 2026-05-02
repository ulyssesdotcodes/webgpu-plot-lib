// Parallel reduction that fills the SeriesBuffer header's extent slots
// (xMin, xMax, yMin, yMax) on the GPU. WGSL has no atomic<f32>, so we map
// each f32 to a u32 with a monotonic ordering, then use atomicMin / atomicMax
// on u32. The mapping:
//
//   positive f32:  bits XOR 0x80000000   (top bit set, ordering preserved)
//   negative f32:  bits XOR 0xFFFFFFFF   (bitwise NOT — flips ordering and
//                                          clears the top bit so all negatives
//                                          live below all positives)
//
// After reduction, the host inverts the same mapping to recover the f32.
// Sentinels:
//   min slots are seeded with encode(+Inf) = 0xFF800000
//   max slots are seeded with encode(-Inf) = 0x007FFFFF

struct Params {
  pointCount:  u32,
  seriesCount: u32,
};

@group(0) @binding(0) var<storage, read> xs: array<f32>;
@group(0) @binding(1) var<storage, read> ys: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<atomic<u32>, 4>;
@group(0) @binding(3) var<uniform> params: Params;

fn encode(f: f32) -> u32 {
  let u = bitcast<u32>(f);
  // If the sign bit is set (negative), flip every bit; otherwise toggle just
  // the sign bit so positives sort above negatives.
  if (u & 0x80000000u) != 0u { return ~u; }
  return u ^ 0x80000000u;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let total = params.pointCount * params.seriesCount;
  let i = gid.x;
  if i >= total { return; }

  let series = i / params.pointCount;
  let pt     = i % params.pointCount;

  let yEnc = encode(ys[series * params.pointCount + pt]);
  atomicMin(&result[2], yEnc);
  atomicMax(&result[3], yEnc);

  // X is shared across series, so only one thread per X point participates.
  if series == 0u {
    let xEnc = encode(xs[pt]);
    atomicMin(&result[0], xEnc);
    atomicMax(&result[1], xEnc);
  }
}
