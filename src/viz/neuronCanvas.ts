/** Neuron state renderer: fixed grid of circles, fill = activation, flash = write. */

export interface NeuronRenderOpts {
  maxAbs?: number;      // normalization
  flash?: Float32Array; // optional per-neuron write magnitude this step
}

export function renderNeurons(canvas: HTMLCanvasElement, x: Float32Array, opts: NeuronRenderOpts = {}): void {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
    canvas.width = w * dpr; canvas.height = h * dpr;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const n = x.length;
  let maxAbs = opts.maxAbs ?? 0;
  if (opts.maxAbs === undefined) {
    for (let i = 0; i < n; i++) maxAbs = Math.max(maxAbs, Math.abs(x[i]));
    maxAbs = maxAbs || 1;
  }
  const cols = Math.ceil(Math.sqrt(n * (w / Math.max(h, 1))));
  const rows = Math.ceil(n / cols);
  const cell = Math.min(w / cols, h / rows);
  const r = Math.max(2, cell * 0.36);

  for (let i = 0; i < n; i++) {
    const cx = (i % cols) * cell + cell / 2;
    const cy = Math.floor(i / cols) * cell + cell / 2;
    const a = x[i] / maxAbs;
    if (a > 0) {
      ctx.fillStyle = `rgba(96, 220, 160, ${0.12 + 0.88 * Math.min(1, a)})`;
    } else {
      ctx.fillStyle = `rgba(255, 118, 118, ${0.12 + 0.88 * Math.min(1, -a)})`;
    }
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fill();
    const flash = opts.flash?.[i] ?? 0;
    if (Math.abs(flash) > 1e-6) {
      const f = Math.min(1, Math.abs(flash) / (maxAbs || 1));
      ctx.strokeStyle = `rgba(255, 214, 102, ${0.3 + 0.7 * f})`;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(cx, cy, r + 2 + 3 * f, 0, Math.PI * 2);
      ctx.stroke();
    }
  }
}
