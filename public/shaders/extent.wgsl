// Compute shader — finds Y min/max per axis over the visible X window.
// Dispatch: dispatchWorkgroups(axisCount) — each workgroup handles one axis.
// Result layout: [yMin0, yMax0, yMin1, yMax1, ...]

struct ExtentParams {
  xMin:        f32,
  xMax:        f32,
  seriesCount: u32,
  pointCount:  u32,
};

@group(0) @binding(0) var<uniform>            params:     ExtentParams;
@group(0) @binding(1) var<storage,read>       xs:         array<f32>;
@group(0) @binding(2) var<storage,read>       ys:         array<f32>;
@group(0) @binding(3) var<storage,read_write> result:     array<f32>;  // axisCount * 2
@group(0) @binding(4) var<storage,read>       seriesAxis: array<u32>;  // seriesCount

const WG:  u32 = 256u;
const INF: f32 = 1e38;

var<workgroup> sMin: array<f32, WG>;
var<workgroup> sMax: array<f32, WG>;

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let axisIdx  = wid.x;
  let tid      = lid.x;
  let totalY   = params.seriesCount * params.pointCount;

  var localMin: f32 =  INF;
  var localMax: f32 = -INF;

  for (var i = tid; i < totalY; i += WG) {
    let ser = i / params.pointCount;
    if seriesAxis[ser] != axisIdx { continue; }
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
    result[axisIdx * 2u]      = sMin[0];
    result[axisIdx * 2u + 1u] = sMax[0];
  }
}
