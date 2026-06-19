export { LineChart } from './chart.js';
export type { HoverInfo, LineChartOptions } from './chart.js';
export {
  createSeriesBuffer,
  viewSeriesBuffer,
  setStyle,
  setAxisConfig,
  recomputeExtents,
  getExtents,
  getAxisCpuExtents,
  getSeriesAxis,
  getAxisColor,
  MAGIC,
  VERSION,
  HEADER_BYTES,
  META_STRIDE,
  AXIS_STRIDE,
  FLAG_SHARED_X,
} from './format.js';
export type {
  PointShape,
  PointStyle,
  SeriesStyle,
  AxisConfig,
  SeriesBufferView,
} from './format.js';
export { generateLines, generateClusters } from './data.js';
