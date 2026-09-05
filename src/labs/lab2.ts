/** Lab 2: validate a recurrent linear-attention state against its history sum. */
import {
  CausalLinearAttentionState,
  ExplicitHistoryAttention,
  softmaxAttentionWeights,
} from '../engine/attention.js';

const KEY_DIMENSION = 12;
const VALUE_DIMENSION = 8;
const SEQ = [1, 2, 3, 4, 5, 6, 2, 3, 1, 2, 3, 4, 7, 8, 2, 3];
const LABELS: Record<number, string> = {
  1: 'the', 2: 'cat', 3: 'sat', 4: 'on', 5: 'a', 6: 'mat', 7: 'dog', 8: 'ran',
};

function vectorFor(token: number, dimension: number, salt: number): Float32Array {
  let state = (token * 0x9e3779b1 + salt) >>> 0;
  const vector = new Float32Array(dimension);
  let norm = 0;
  for (let i = 0; i < dimension; i++) {
    state = (Math.imul(state ^ (state >>> 16), 0x21f0aaad) + i + 1) >>> 0;
    const value = (state / 0xffffffff) * 2 - 1;
    vector[i] = value;
    norm += value * value;
  }
  norm = Math.sqrt(norm) || 1;
  for (let i = 0; i < dimension; i++) vector[i] /= norm;
  return vector;
}

function maxAbsoluteError(a: Float32Array, b: Float32Array): number {
  let max = 0;
  for (let i = 0; i < a.length; i++) max = Math.max(max, Math.abs(a[i] - b[i]));
  return max;
}

export function initLab2(): void {
  const querySlider = document.getElementById('lab2-query') as HTMLInputElement | null;
  const gammaSlider = document.getElementById('lab2-gamma') as HTMLInputElement | null;
  const gammaVal = document.getElementById('lab2-gamma-val');
  const queryVal = document.getElementById('lab2-query-val');
  const metric = document.getElementById('lab2-corr');
  const canvas = document.getElementById('lab2-bars') as HTMLCanvasElement | null;
  const tokensEl = document.getElementById('lab2-tokens');
  if (!querySlider || !gammaSlider || !gammaVal || !queryVal || !metric || !canvas || !tokensEl) {
    console.warn('Lab 2 was not initialized because its page controls are incomplete.');
    return;
  }

  const queryControl = querySlider;
  const decayControl = gammaSlider;
  const metricReadout = metric;
  const outputCanvas = canvas;
  const tokenStrip = tokensEl;
  outputCanvas.setAttribute('aria-label', 'Recurrent and explicit-history linear-attention outputs with a separate softmax contrast');

  const readout = metric.closest('.readout');
  readout?.setAttribute('role', 'status');
  readout?.setAttribute('aria-live', 'polite');
  readout?.setAttribute('aria-atomic', 'true');
  if (readout?.firstChild) readout.firstChild.textContent = 'recurrent vs explicit-history check: ';
  const hint = readout?.querySelector('.hint');
  if (hint) hint.textContent = 'The first two outputs should agree up to floating-point rounding. Softmax is a separate contrast and is not expected to match.';

  const intro = document.querySelector('#lab2 .lab-head p');
  if (intro) {
    intro.innerHTML = `We compare the actual fixed <span class="eq">${KEY_DIMENSION}×${VALUE_DIMENSION}</span> recurrent state read with a direct sum over every prior key/value pair. Those two are equivalent implementations of <strong>causal decayed linear attention</strong>. Softmax uses a normalized exponential over history and is shown only as a contrast—not as an equivalent output.`;
  }
  const insight = document.getElementById('tour-2');
  if (insight) {
    insight.innerHTML = '<strong>The lesson:</strong> recurrence and explicit history can compute the same unnormalized linear-attention output. Their storage costs differ, but the equality is directly checked here. The separately labeled softmax output generally differs because normalization and exponential weighting change the operation.';
  }

  tokensEl.replaceChildren(...SEQ.map((token, index) => {
    const item = document.createElement('span');
    item.className = 'tok';
    item.dataset.i = String(index);
    item.title = `position ${index}`;
    item.textContent = LABELS[token] ?? String(token);
    return item;
  }));

  function run(): void {
    const decay = +decayControl.value;
    const t = Math.min(SEQ.length - 1, Math.max(0, +queryControl.value));
    const keys = SEQ.map((token) => vectorFor(token, KEY_DIMENSION, 17));
    const values = SEQ.map((token) => vectorFor(token, VALUE_DIMENSION, 53));
    const recurrent = new CausalLinearAttentionState(KEY_DIMENSION, VALUE_DIMENSION, decay);
    const history = new ExplicitHistoryAttention(decay);
    let recurrentRead: Float32Array = new Float32Array(VALUE_DIMENSION);
    let historyRead: Float32Array = new Float32Array(VALUE_DIMENSION);

    for (let i = 0; i <= t; i++) {
      recurrentRead = recurrent.step(keys[i], keys[i], values[i]).read;
      historyRead = history.step(keys[i], keys[i], values[i]);
    }

    const softmaxWeights = softmaxAttentionWeights(keys, t);
    const softmaxRead = new Float32Array(VALUE_DIMENSION);
    for (let tau = 0; tau < t; tau++) {
      for (let j = 0; j < VALUE_DIMENSION; j++) {
        softmaxRead[j] += softmaxWeights[tau] * values[tau][j];
      }
    }

    const error = maxAbsoluteError(recurrentRead, historyRead);
    metricReadout.textContent = `max |recurrent − history| = ${error.toExponential(2)}`;
    metricReadout.style.color = error < 1e-5 ? '#7fe0b0' : '#ff7d6e';
    drawOutputs(outputCanvas, recurrentRead, historyRead, softmaxRead);
    tokenStrip.querySelectorAll('.tok').forEach((element, index) => {
      element.classList.toggle('tok-query', index === t);
      element.classList.toggle('tok-past', index < t);
    });
  }

  querySlider.addEventListener('input', () => {
    queryVal.textContent = querySlider.value;
    run();
  });
  gammaSlider.addEventListener('input', () => {
    gammaVal.textContent = (+gammaSlider.value).toFixed(2);
    run();
  });

  queryVal.textContent = querySlider.value;
  gammaVal.textContent = (+gammaSlider.value).toFixed(2);
  run();
}

function drawOutputs(
  canvas: HTMLCanvasElement,
  recurrent: Float32Array,
  explicitHistory: Float32Array,
  softmaxContrast: Float32Array,
): void {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (canvas.width !== width * dpr || canvas.height !== height * dpr) {
    canvas.width = width * dpr;
    canvas.height = height * dpr;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  const top = 34;
  const bottom = 28;
  const mid = top + (height - top - bottom) / 2;
  const magnitude = Math.max(0.001, ...Array.from(recurrent, Math.abs), ...Array.from(softmaxContrast, Math.abs));
  const componentWidth = width / VALUE_DIMENSION;
  const scale = (height - top - bottom) / 2 / magnitude;
  const series = [
    { values: recurrent, color: 'rgba(96,220,160,0.9)', offset: -componentWidth * 0.26 },
    { values: explicitHistory, color: 'rgba(110,170,255,0.75)', offset: 0 },
    { values: softmaxContrast, color: 'rgba(255,159,110,0.8)', offset: componentWidth * 0.26 },
  ];

  ctx.strokeStyle = 'rgba(255,255,255,0.25)';
  ctx.beginPath();
  ctx.moveTo(0, mid);
  ctx.lineTo(width, mid);
  ctx.stroke();
  for (const { values, color, offset } of series) {
    ctx.fillStyle = color;
    for (let j = 0; j < VALUE_DIMENSION; j++) {
      const barHeight = values[j] * scale;
      const x = (j + 0.5) * componentWidth + offset - componentWidth * 0.1;
      ctx.fillRect(x, mid - Math.max(0, barHeight), componentWidth * 0.2, Math.abs(barHeight));
    }
  }
  ctx.font = '11px system-ui, sans-serif';
  ctx.fillStyle = 'rgba(96,220,160,0.95)';
  ctx.fillText('recurrent state output', 6, 12);
  ctx.fillStyle = 'rgba(110,170,255,0.95)';
  ctx.fillText('explicit-history linear reference', 148, 12);
  ctx.fillStyle = 'rgba(255,159,110,0.95)';
  ctx.fillText('softmax contrast (not equivalent)', 355, 12);
  ctx.fillStyle = 'rgba(255,255,255,0.6)';
  for (let j = 0; j < VALUE_DIMENSION; j++) ctx.fillText(String(j), (j + 0.5) * componentWidth - 3, height - 7);
}
