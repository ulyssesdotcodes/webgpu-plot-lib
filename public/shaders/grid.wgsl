// Full-screen pass that paints axis grid lines. gridStep lives in the View
// uniform (written by the chart on zoom/resize) so the grid shader needs no
// separate grid-step upload.  Immutable style (color + bg) lives in a tiny
// uniform written once at init.

struct View {
  scale:       vec2f,
  offset:      vec2f,
  viewport:    vec2f,
  gridStep:    vec2f,
  pointCount:  u32,
  seriesCount: u32,
};

struct GridStyle {
  color: vec4f,
  bg:    vec4f,
};

@group(0) @binding(0) var<uniform> view:  View;
@group(0) @binding(1) var<uniform> style: GridStyle;

const TRI = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));

@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  return vec4f(TRI[vi], 0.0, 1.0);
}

@fragment fn fs(@builtin(position) frag: vec4f) -> @location(0) vec4f {
  let ndc = vec2f(
    (frag.x / view.viewport.x) * 2.0 - 1.0,
    1.0 - (frag.y / view.viewport.y) * 2.0,
  );
  let data = ndc / view.scale + view.offset;

  let g = abs(fract(data / view.gridStep + vec2f(0.5)) - vec2f(0.5)) * view.gridStep;

  let dataPerPx = abs(vec2f(2.0) / (view.scale * view.viewport));
  let halfLine  = dataPerPx * 0.5;

  let mx = 1.0 - smoothstep(halfLine.x, halfLine.x + dataPerPx.x, g.x);
  let my = 1.0 - smoothstep(halfLine.y, halfLine.y + dataPerPx.y, g.y);
  let m  = max(mx, my);

  return vec4f(mix(style.bg.rgb, style.color.rgb, m * style.color.a), 1.0);
}
