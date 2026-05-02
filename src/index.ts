import { LineChart, type HoverInfo } from './chart.js';
import { generateLines } from './data.js';

function fail(msg: string): never {
  document.body.innerHTML = `<pre style="color:red;padding:1em">${msg}</pre>`;
  throw new Error(msg);
}

async function main(): Promise<void> {
  const canvas = document.querySelector('canvas');
  if (!canvas) fail('no canvas element');

  const tooltip = document.getElementById('tooltip');
  if (!tooltip) fail('no #tooltip element');

  const data = generateLines(6, 4096);

  const chart = await LineChart.create(canvas, data, {
    onHover: (info: HoverInfo | null) => {
      if (!info) {
        tooltip.style.display = 'none';
        return;
      }
      tooltip.style.display = 'block';
      tooltip.innerHTML =
        `<b>series ${info.series}</b> · point ${info.point}<br>` +
        `x = ${info.x.toFixed(4)}<br>y = ${info.y.toFixed(4)}`;
    },
  });

  // Position the tooltip from the same mousemove the chart sees.
  canvas.addEventListener('mousemove', (e) => {
    tooltip.style.left = `${e.clientX + 12}px`;
    tooltip.style.top  = `${e.clientY + 12}px`;
  });

  // Expose a couple of handles for poking at in the console.
  (window as unknown as { chart: LineChart; data: typeof data }).chart = chart;
  (window as unknown as { chart: LineChart; data: typeof data }).data = data;
}

main();
