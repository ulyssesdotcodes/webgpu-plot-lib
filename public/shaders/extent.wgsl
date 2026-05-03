// Compute shader — finds Y min/max over the visible X window.
// Single workgroup of 256 threads; stride-loop handles any seriesCount×pointCount.
// Dispatch: always 1 workgroup.
//
// If no points fall in [xMin, xMax], result[0] = +Inf, result[1] = -Inf.

struct ExtentParams {
  xMin:       f32,
  xMax:       f32,
  totalY:     u32,  // = seriesCount * pointCount
  pointCount: u32,
};

@group(0) @binding(0) var<uniform>            params: ExtentParams;
@group(0) @binding(1) var<storage,read>       xs:     array<f32>;
@group(0) @binding(2) var<storage,read>       ys:     array<f32>;
@group(0) @binding(3) var<storage,read_write> result: array<f32, 2>;  // [yMin, yMax]

const WG:  u32 = 256u;
const INF: f32 = 1e38;

var<workgroup> sMin: array<f32, WG>;
var<workgroup> sMax: array<f32, WG>;

@compute @workgroup_size(WG)
fn main(@builtin(local_invocation_id) lid: vec3u) {
  let tid = lid.x;

  var localMin: f32 =  INF;
  var localMax: f32 = -INF;

  for (var i = tid; i < params.totalY; i += WG) {
    let idx = i % params.pointCount;
    let x   = xs[idx];
    if x >= params.xMin && x <= params.xMax {
      let y    = ys[i];
      localMin = min(localMin, y);
      localMax = max(localMax, y);
    }
  }

  sMin[tid] = localMin;
  sMax[tid] = localMax;
  workgroupBarrier();

  var s = WG >> 1u;
  loop {
    if s == 0u { break; }
    if tid < s {
      sMin[tid] = min(sMin[tid], sMin[tid + s]);
      sMax[tid] = max(sMax[tid], sMax[tid + s]);
    }
    workgroupBarrier();
    s >>= 1u;
  }

  if tid == 0u {
    result[0] = sMin[0];
    result[1] = sMax[0];
  }
}
