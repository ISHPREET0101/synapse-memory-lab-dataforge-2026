/** Lab 3: empirical nearest-neighbor retrieval from a toy outer-product memory. */
import { mulberry32 } from '../engine/retrieval.js';
import { renderChart } from '../viz/chart.js';

interface RetrievalPoint {
  load: number;
  accuracy: number;
  maxWrongCosine: number;
}

function randomUnit(random: () => number, dimension: number): Float32Array {
  const vector = new Float32Array(dimension);
  let squaredNorm = 0;
  for (let i = 0; i < dimension; i++) {
    vector[i] = random() * 2 - 1;
    squaredNorm += vector[i] * vector[i];
  }
  const norm = Math.sqrt(squaredNorm) || 1;
  for (let i = 0; i < dimension; i++) vector[i] /= norm;
  return vector;
}

function cosine(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA * normB) + 1e-12);
}

function runNearestNeighborTrial(dimension: number, load: number, seed = 42): RetrievalPoint {
  if (load === 0) return { load, accuracy: 1, maxWrongCosine: 0 };
  const random = mulberry32(seed * 100003 + load);
  const keys: Float32Array[] = [];
  const values: Float32Array[] = [];
  const state = new Float32Array(dimension * dimension);
  for (let item = 0; item < load; item++) {
    const key = randomUnit(random, dimension);
    const value = randomUnit(random, dimension);
    keys.push(key);
    values.push(value);
    for (let row = 0; row < dimension; row++) {
      for (let column = 0; column < dimension; column++) {
        state[row * dimension + column] += key[row] * value[column];
      }
    }
  }

  let correct = 0;
  let maxWrongCosine = 0;
  for (let item = 0; item < load; item++) {
    const read = new Float32Array(dimension);
    for (let column = 0; column < dimension; column++) {
      let sum = 0;
      for (let row = 0; row < dimension; row++) sum += keys[item][row] * state[row * dimension + column];
      read[column] = sum;
    }
    let nearestIndex = -1;
    let nearestCosine = -Infinity;
    for (let candidate = 0; candidate < load; candidate++) {
      const similarity = cosine(read, values[candidate]);
      if (candidate !== item) maxWrongCosine = Math.max(maxWrongCosine, similarity);
      if (similarity > nearestCosine) {
        nearestCosine = similarity;
        nearestIndex = candidate;
      }
    }
    if (nearestIndex === item) correct++;
  }
  return { load, accuracy: correct / load, maxWrongCosine };
}

function averageTrial(dimension: number, load: number): RetrievalPoint {
  const trials = [42, 59, 76].map((seed) => runNearestNeighborTrial(dimension, load, seed));
  return {
    load,
    accuracy: trials.reduce((sum, point) => sum + point.accuracy, 0) / trials.length,
    maxWrongCosine: Math.max(...trials.map((point) => point.maxWrongCosine)),
  };
}

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
  chartCanvas.setAttribute('aria-label', 'Nearest-neighbor retrieval accuracy and maximum wrong-value cosine by association load');

  cueOut.setAttribute('role', 'status');
  cueOut.setAttribute('aria-live', 'polite');
  cueOut.setAttribute('aria-atomic', 'true');
  const intro = document.querySelector('#lab3 .lab-head p');
  if (intro) {
    intro.innerHTML = 'This controlled toy stores random key→value associations in one fixed outer-product matrix. A cue counts as correct only when its intended value is the <strong>nearest stored value by cosine similarity</strong>. The maximum cosine to any wrong value exposes the strongest observed collision.';
  }
  const insight = document.getElementById('tour-3');
  if (insight) {
    insight.innerHTML = '<strong>What to notice:</strong> increasing load can increase collisions in this random linear associative-memory experiment. The curve is empirical and seed-dependent; it demonstrates interference but does <strong>not</strong> establish a capacity limit for BDH or any trained model.';
  }
  const caption = canvas.closest('figure')?.querySelector('figcaption');
  if (caption?.firstChild) caption.firstChild.textContent = 'Nearest-neighbor accuracy (green) and maximum wrong cosine (dashed red) ';

  let points: RetrievalPoint[] = [];

  function recomputeCurve(): void {
    const dimension = +dimensionControl.value;
    const selectedLoad = +loadControl.value;
    const xMax = Math.max(+loadControl.max, dimension * 2);
    const step = Math.max(1, Math.round(xMax / 16));
    const loads = new Set<number>([0, selectedLoad, xMax]);
    for (let load = step; load < xMax; load += step) loads.add(load);
    points = [...loads].sort((a, b) => a - b).map((load) => averageTrial(dimension, load));
    draw(dimension, selectedLoad, xMax);
  }

  function draw(dimension: number, selectedLoad: number, xMax: number): void {
    renderChart(chartCanvas, [
      {
        label: 'nearest-neighbor retrieval accuracy (live)',
        color: '#6ee0b0',
        points: points.map((point) => ({ x: point.load, y: point.accuracy })),
      },
      {
        label: 'maximum wrong-value cosine (live)',
        color: '#ff7d6e',
        dashed: true,
        points: points.map((point) => ({ x: point.load, y: point.maxWrongCosine })),
      },
    ], { xLabel: 'associations stored', yLabel: 'empirical score', xMax, yMin: 0, yMax: 1.05 });
    highlightSelectedLoad(chartCanvas, selectedLoad, xMax);
    const selected = points.find((point) => point.load === selectedLoad);
    if (selected) announce(dimension, selected);
  }

  function announce(dimension: number, selected: RetrievalPoint): void {
    resultReadout.replaceChildren();
    const strong = document.createElement('strong');
    strong.textContent = `Selected load ${selected.load}`;
    resultReadout.append(
      strong,
      ` in a ${dimension}×${dimension} toy state: nearest-neighbor accuracy ${(selected.accuracy * 100).toFixed(1)}%; maximum wrong cosine ${selected.maxWrongCosine.toFixed(3)}. Empirical random-association result—not a BDH capacity claim.`,
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
