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

  const data = generateLines([3, 2, 2], 4096 * 8);

  const chart = await LineChart.create(canvas, data, {
    onHover: (info: HoverInfo | null) => {
      if (!info) { tooltip.style.display = 'none'; return; }
      const lines = [`<b>x = ${info.x.toFixed(4)}</b>`];
      for (let s = 0; s < info.ys.length; s++) {
        const r = Math.round(data.meta[s * 12 + 0]! * 255);
        const g = Math.round(data.meta[s * 12 + 1]! * 255);
        const b = Math.round(data.meta[s * 12 + 2]! * 255);
        lines.push(`<span style="color:rgb(${r},${g},${b})">■</span> ${info.ys[s]!.toFixed(4)}`);
      }
      tooltip.style.display = 'block';
      tooltip.innerHTML = lines.join('<br>');
    },
  });

  canvas.addEventListener('mousemove', (e) => {
    tooltip.style.left = `${e.clientX + 12}px`;
    tooltip.style.top  = `${e.clientY + 12}px`;
  });

  (window as unknown as { chart: LineChart; data: typeof data }).chart = chart;
  (window as unknown as { chart: LineChart; data: typeof data }).data  = data;
}

main();
