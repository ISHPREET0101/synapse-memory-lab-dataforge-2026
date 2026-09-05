/** Synapse matrix renderer: N×N heat view. Positive = green, negative = red, white flash on writes. */

export function renderSynapses(canvas: HTMLCanvasElement, S: Float32Array, n: number, maxAbs?: number): void {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const dpr = window.devicePixelRatio || 1;
  const size = Math.min(canvas.clientWidth, canvas.clientHeight);
  if (canvas.width !== size * dpr || canvas.height !== size * dpr) {
    canvas.width = size * dpr; canvas.height = size * dpr;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, size, size);

  let m = maxAbs ?? 0;
  if (maxAbs === undefined) {
    for (let i = 0; i < S.length; i++) m = Math.max(m, Math.abs(S[i]));
    m = m || 1;
  }
  const px = Math.max(1, Math.floor(size / n));
  const img = ctx.createImageData(n, n);
  const data = img.data;
  for (let i = 0; i < n * n; i++) {
    const a = S[i] / m;
    const p = i * 4;
    if (a >= 0) {
      data[p] = 40 + 60 * a; data[p + 1] = 60 + 175 * a; data[p + 2] = 80 + 90 * a;
    } else {
      data[p] = 90 + 150 * -a; data[p + 1] = 55 + 55 * -a; data[p + 2] = 70 + 40 * -a;
    }
    data[p + 3] = 255;
  }
  // draw scaled
  const tmp = document.createElement('canvas');
  tmp.width = n; tmp.height = n;
  tmp.getContext('2d')!.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(tmp, 0, 0, n, n, 0, 0, px * n, px * n);
}
