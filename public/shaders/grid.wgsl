// Full-screen pass that paints axis grid lines at every `grid.step` data unit.
// Drawn before the lines pass so it sits behind everything. The chart computes
// "nice" tick spacing on the CPU and feeds it in via the Grid uniform.

struct View {
  scale:        vec2f,
  offset:       vec2f,
  viewport:     vec2f,
  pointCount:   u32,
  seriesCount:  u32,
};

struct Grid {
  step:  vec2f,
  color: vec4f,
  bg:    vec4f,
};

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<uniform> grid: Grid;

const TRI = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));

@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  return vec4f(TRI[vi], 0.0, 1.0);
}

@fragment fn fs(@builtin(position) frag: vec4f) -> @location(0) vec4f {
  // Pixel position → NDC → data domain.
  let ndc = vec2f(
    (frag.x / view.viewport.x) * 2.0 - 1.0,
    1.0 - (frag.y / view.viewport.y) * 2.0,
  );
  let data = ndc / view.scale + view.offset;

  // Distance, in data units, from this fragment to the nearest grid line on
  // each axis. fract(d/step + 0.5) - 0.5 ranges over [-0.5, 0.5]; abs * step
  // gives the data-space distance to the closest gridline.
  let g = abs(fract(data / grid.step + vec2f(0.5)) - vec2f(0.5)) * grid.step;

  // Convert "1 pixel" into data units per axis; sign of scale.y is flipped, so
  // take abs.
  let dataPerPx = abs(vec2f(2.0) / (view.scale * view.viewport));
  let halfLine  = dataPerPx * 0.5;

  let mx = 1.0 - smoothstep(halfLine.x, halfLine.x + dataPerPx.x, g.x);
  let my = 1.0 - smoothstep(halfLine.y, halfLine.y + dataPerPx.y, g.y);
  let m  = max(mx, my);

  return vec4f(mix(grid.bg.rgb, grid.color.rgb, m * grid.color.a), 1.0);
}
