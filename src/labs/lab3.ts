/** Lab 3: empirical nearest-neighbor retrieval from a toy outer-product memory. */
import { runCapacityExperiment, type CapacityPoint } from '../engine/retrieval.js';
import { renderChart } from '../viz/chart.js';

export function initLab3(): void {
  const loadSlider = document.getElementById('lab3-load') as HTMLInputElement | null;
  const loadVal = document.getElementById('lab3-load-val');
  const dimSel = document.getElementById('lab3-dim') as HTMLSelectElement | null;
  const cueBtn = document.getElementById('lab3-cue') as HTMLButtonElement | null;
  const cueOut = document.getElementById('lab3-cue-out');
  const canvas = document.getElementById('lab3-chart') as HTMLCanvasElement | null;
  if (!loadSlider || !loadVal || !dimSel || !cueBtn || !cueOut || !canvas) {
    console.warn('Lab 3 was not initialized because its page controls are incomplete.');
    return;
  }

  const loadControl = loadSlider;
  const dimensionControl = dimSel;
  const resultReadout = cueOut;
  const chartCanvas = canvas;
  chartCanvas.setAttribute('aria-label', 'Nearest-neighbor retrieval accuracy and mean strongest wrong-value cosine by association load');

  cueOut.setAttribute('role', 'status');
  cueOut.setAttribute('aria-live', 'polite');
  cueOut.setAttribute('aria-atomic', 'true');
  const intro = document.querySelector('#lab3 .lab-head p');
  if (intro) {
    intro.innerHTML = 'This controlled toy stores random key→value associations in one fixed outer-product matrix. A cue counts as correct only when its intended value is the <strong>nearest stored value by cosine similarity</strong>. The mean strongest wrong-value cosine exposes the typical closest collision.';
  }
  const insight = document.getElementById('tour-3');
  if (insight) {
    insight.innerHTML = '<strong>What to notice:</strong> increasing load can increase collisions in this random linear associative-memory experiment. The curve is empirical and seed-dependent; it demonstrates interference but does <strong>not</strong> establish a capacity limit for BDH or any trained model.';
  }
  const caption = canvas.closest('figure')?.querySelector('figcaption');
  if (caption?.firstChild) caption.firstChild.textContent = 'Nearest-neighbor accuracy (green) and mean strongest wrong-value cosine (dashed red) ';

  let points: CapacityPoint[] = [];

  function recomputeCurve(): void {
    const dimension = +dimensionControl.value;
    const selectedLoad = +loadControl.value;
    const xMax = Math.max(+loadControl.max, dimension * 2);
    const step = Math.max(1, Math.round(xMax / 16));
    const loads = new Set<number>([0, selectedLoad, xMax]);
    for (let load = step; load < xMax; load += step) loads.add(load);
    points = runCapacityExperiment(dimension, [...loads].sort((a, b) => a - b), 42);
    draw(dimension, selectedLoad, xMax);
  }

  function draw(dimension: number, selectedLoad: number, xMax: number): void {
    renderChart(chartCanvas, [
      {
        label: 'nearest-neighbor retrieval accuracy (live)',
        color: '#6ee0b0',
        points: points.map((point) => ({ x: point.load, y: point.recall })),
      },
      {
        label: 'mean strongest wrong-value cosine (live)',
        color: '#ff7d6e',
        dashed: true,
        points: points.map((point) => ({ x: point.load, y: point.maxWrongCosine })),
      },
    ], { xLabel: 'associations stored', yLabel: 'empirical score', xMax, yMin: 0, yMax: 1.05 });
    highlightSelectedLoad(chartCanvas, selectedLoad, xMax);
    const selected = points.find((point) => point.load === selectedLoad);
    if (selected) announce(dimension, selected);
  }

  function announce(dimension: number, selected: CapacityPoint): void {
    resultReadout.replaceChildren();
    const strong = document.createElement('strong');
    strong.textContent = `Selected load ${selected.load}`;
    resultReadout.append(
      strong,
      ` in a ${dimension}×${dimension} toy state: nearest-neighbor accuracy ${(selected.recall * 100).toFixed(1)}%; mean strongest wrong-value cosine ${selected.maxWrongCosine.toFixed(3)}. Empirical random-association result—not a BDH capacity claim.`,
    );
  }

  loadSlider.addEventListener('input', () => {
    loadVal.textContent = loadSlider.value;
    recomputeCurve();
  });
  dimSel.addEventListener('change', recomputeCurve);
  cueBtn.addEventListener('click', recomputeCurve);

  loadVal.textContent = loadSlider.value;
  recomputeCurve();
}

function highlightSelectedLoad(canvas: HTMLCanvasElement, load: number, xMax: number): void {
  const context = canvas.getContext('2d');
  if (!context || xMax <= 0) return;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  const x = 46 + (load / xMax) * (width - 46 - 12);
  context.save();
  context.strokeStyle = '#ffd66e';
  context.fillStyle = '#ffd66e';
  context.lineWidth = 2;
  context.setLineDash([3, 3]);
  context.beginPath();
  context.moveTo(x, 12);
  context.lineTo(x, height - 34);
  context.stroke();
  context.setLineDash([]);
  context.font = 'bold 11px system-ui, sans-serif';
  context.fillText(`selected: ${load}`, Math.min(x + 5, width - 82), 24);
  context.restore();
}
