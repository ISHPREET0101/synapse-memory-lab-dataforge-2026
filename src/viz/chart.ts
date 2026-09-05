/** Minimal dependency-free line/scatter chart for capacity curves and comparisons. */

export interface Series {
  label: string;
  points: { x: number; y: number }[];
  color: string;
  dashed?: boolean;
  live?: boolean; // rendered marker legend: "live" vs "precomputed"
}

export interface ChartOpts {
  xLabel: string;
  yLabel: string;
  yMin?: number;
  yMax?: number;
  xMax?: number;
}

export function renderChart(canvas: HTMLCanvasElement, series: Series[], opts: ChartOpts): void {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
    canvas.width = w * dpr; canvas.height = h * dpr;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const padL = 46, padB = 34, padT = 12, padR = 12;
  const iw = w - padL - padR, ih = h - padT - padB;
  const all = series.flatMap((s) => s.points);
  if (!all.length) return;
  const xMax = opts.xMax ?? Math.max(...all.map((p) => p.x));
  const yMin = opts.yMin ?? 0;
  const yMax = opts.yMax ?? Math.max(...all.map((p) => p.y)) * 1.05;
  const X = (x: number) => padL + (x / xMax) * iw;
  const Y = (y: number) => padT + ih - ((y - yMin) / (yMax - yMin || 1)) * ih;

  ctx.strokeStyle = 'rgba(255,255,255,0.12)';
  ctx.fillStyle = 'rgba(255,255,255,0.55)';
  ctx.font = '11px system-ui, sans-serif';
  ctx.lineWidth = 1;
  for (let g = 0; g <= 4; g++) {
    const y = yMin + ((yMax - yMin) * g) / 4;
    ctx.beginPath(); ctx.moveTo(padL, Y(y)); ctx.lineTo(w - padR, Y(y)); ctx.stroke();
    ctx.fillText(y.toFixed(2), 4, Y(y) + 4);
  }
  for (let g = 0; g <= 4; g++) {
    const x = (xMax * g) / 4;
    ctx.fillText(String(Math.round(x)), X(x) - 6, h - 16);
  }
  ctx.fillText(opts.xLabel, w / 2 - 30, h - 2);
  ctx.save();
  ctx.translate(10, h / 2 + 30); ctx.rotate(-Math.PI / 2);
  ctx.fillText(opts.yLabel, 0, 0);
  ctx.restore();

  for (const s of series) {
    if (!s.points.length) continue;
    ctx.strokeStyle = s.color;
    ctx.setLineDash(s.dashed ? [5, 4] : []);
    ctx.lineWidth = 2;
    ctx.beginPath();
    s.points.forEach((p, i) => (i ? ctx.lineTo(X(p.x), Y(p.y)) : ctx.moveTo(X(p.x), Y(p.y))));
    ctx.stroke();
    ctx.setLineDash([]);
    for (const p of s.points) {
      ctx.fillStyle = s.color;
      ctx.beginPath(); ctx.arc(X(p.x), Y(p.y), 3, 0, Math.PI * 2); ctx.fill();
    }
  }
}
