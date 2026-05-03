import { LineChart } from './chart.js';
import { generateLines } from './data.js';

function fail(msg: string): never {
  document.body.innerHTML = `<pre style="color:red;padding:1em">${msg}</pre>`;
  throw new Error(msg);
}

async function main(): Promise<void> {
  const canvas = document.querySelector('canvas');
  if (!canvas) fail('no canvas element');

  const data = generateLines(6, 4096);

  const chart = await LineChart.create(canvas, data);

  (window as unknown as { chart: LineChart; data: typeof data }).chart = chart;
  (window as unknown as { chart: LineChart; data: typeof data }).data  = data;
}

main();
