/** Lab 1: inspect a fixed-shape recurrent outer-product state as it changes. */
import { BDHCell } from '../engine/bdh.js';
import { renderNeurons } from '../viz/neuronCanvas.js';
import { renderSynapses } from '../viz/synapseCanvas.js';

const N = 96;
const D = 12;
const STATE_ENTRIES = N * D;
const PRESET_TOKENS = [1, 2, 3, 4, 5, 6, 2, 3, 1, 2, 3, 4, 7, 8, 2, 3, 9, 10, 2, 3, 11, 12, 2, 3];
const TOKEN_LABELS: Record<number, string> = {
  1: 'the', 2: 'cat', 3: 'sat', 4: 'on', 5: 'a', 6: 'mat', 7: 'dog', 8: 'ran',
  9: 'birds', 10: 'sing', 11: 'fish', 12: 'swim',
};

export interface Lab1Controller {
  setRunning: (running: boolean) => void;
}

const inertController: Lab1Controller = { setRunning: () => undefined };

export function initLab1(): Lab1Controller {
  const neuronCanvas = document.getElementById('lab1-neurons') as HTMLCanvasElement | null;
  const synCanvas = document.getElementById('lab1-synapses') as HTMLCanvasElement | null;
  const tokenLabel = document.getElementById('lab1-token');
  const stepCounter = document.getElementById('lab1-steps');
  const sparsity = document.getElementById('lab1-sparsity');
  const stateScale = document.getElementById('lab1-scale') as HTMLInputElement | null;
  const gamma = document.getElementById('lab1-gamma') as HTMLInputElement | null;
  const speed = document.getElementById('lab1-speed') as HTMLInputElement | null;
  const scaleVal = document.getElementById('lab1-scale-val');
  const gammaVal = document.getElementById('lab1-gamma-val');
  const speedVal = document.getElementById('lab1-speed-val');
  const pauseBtn = document.getElementById('lab1-pause') as HTMLButtonElement | null;
  const stepBtn = document.getElementById('lab1-step') as HTMLButtonElement | null;
  const resetBtn = document.getElementById('lab1-reset') as HTMLButtonElement | null;

  if (!neuronCanvas || !synCanvas || !tokenLabel || !stepCounter || !sparsity ||
      !stateScale || !gamma || !speed || !scaleVal || !gammaVal || !speedVal ||
      !pauseBtn || !stepBtn || !resetBtn) {
    console.warn('Lab 1 was not initialized because its page controls are incomplete.');
    return inertController;
  }

  const neuronView = neuronCanvas;
  const synapseView = synCanvas;
  const tokenReadout = tokenLabel;
  const stepReadout = stepCounter;
  const sparsityReadout = sparsity;
  const scaleControl = stateScale;
  const decayControl = gamma;
  const speedControl = speed;

  const status = document.createElement('div');
  status.className = 'readout';
  status.setAttribute('role', 'status');
  status.setAttribute('aria-live', 'polite');
  status.setAttribute('aria-atomic', 'true');
  resetBtn.closest('.controls')?.append(status);
  synapseView.setAttribute('aria-label', `Derived synapse projection from the fixed ${N} by ${D} recurrent state`);

  const intro = document.querySelector('#lab1 .lab-head p');
  if (intro) {
    intro.innerHTML = `Each token reads the previous state, then writes an outer product into <span class="eq">σ ← γσ + x ⊗ v</span>. The recurrent state is always <strong>${N}×${D} = ${STATE_ENTRIES.toLocaleString()} floats</strong>, regardless of sequence length. The neuron view shows x; the square heatmap shows the derived projection σD<sub>y</sub>.`;
  }

  let cell = makeCell();
  let pos = 0;
  let running = true;
  let timer: number | undefined;

  function makeCell(): BDHCell {
    return new BDHCell({ n: N, d: D, scale: +scaleControl.value, gamma: +decayControl.value, seed: 7 });
  }

  function announce(message: string): void {
    status.textContent = `${message} Fixed recurrent state: ${N}×${D} = ${STATE_ENTRIES.toLocaleString()} floats (${(STATE_ENTRIES * Float32Array.BYTES_PER_ELEMENT).toLocaleString()} bytes).`;
  }

  function schedule(): void {
    window.clearTimeout(timer);
    if (!running) return;
    timer = window.setTimeout(tick, 1000 / Math.max(1, +speedControl.value));
  }

  function tick(): void {
    if (!running) return;
    stepOnce();
    schedule();
  }

  function stepOnce(): void {
    const token = PRESET_TOKENS[pos % PRESET_TOKENS.length];
    pos++;
    draw(cell.step(token), token);
  }

  function draw(result: ReturnType<BDHCell['step']>, token: number): void {
    renderNeurons(neuronView, result.x, { flash: flashPerNeuron(result) });
    renderSynapses(synapseView, cell.synapseMatrix(), cell.cfg.n);
    tokenReadout.textContent = TOKEN_LABELS[token] ?? `tok_${token}`;
    stepReadout.textContent = String(cell.stepCount);
    const active = result.activeCount / cell.cfg.n;
    sparsityReadout.textContent = `${(active * 100).toFixed(1)}% active`;
    sparsityReadout.style.color = active > 0.25 ? '#ff9f6e' : '#7fe0b0';
  }

  function flashPerNeuron(result: ReturnType<BDHCell['step']>): Float32Array {
    const out = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      let maxWrite = 0;
      for (let k = 0; k < D; k++) {
        maxWrite = Math.max(maxWrite, Math.abs(result.hebbianWrite[i * D + k]));
      }
      out[i] = maxWrite;
    }
    return out;
  }

  function drawEmptyState(): void {
    renderNeurons(neuronView, new Float32Array(N));
    renderSynapses(synapseView, cell.synapseMatrix(), N);
    tokenReadout.textContent = '—';
    stepReadout.textContent = '0';
    sparsityReadout.textContent = '0.0% active';
  }

  function reset(reason: string): void {
    window.clearTimeout(timer);
    cell = makeCell();
    pos = 0;
    drawEmptyState();
    announce(`${reason}: σ was zero-filled and the step counter returned to 0; prior tokens are not retained.`);
    schedule();
  }

  stateScale.addEventListener('input', () => {
    scaleVal.textContent = stateScale.value;
    reset('Update scale changed');
  });
  gamma.addEventListener('input', () => {
    gammaVal.textContent = (+gamma.value).toFixed(2);
    reset('Decay changed');
  });
  speed.addEventListener('input', () => {
    speedVal.textContent = speed.value;
    announce(`Replay speed set to ${speed.value} tokens per second.`);
    schedule();
  });
  pauseBtn.addEventListener('click', () => {
    running = !running;
    pauseBtn.textContent = running ? 'Pause' : 'Resume';
    announce(running ? 'Replay resumed.' : 'Replay paused; the current state is preserved.');
    schedule();
  });
  stepBtn.addEventListener('click', () => {
    stepOnce();
    announce(`Advanced to step ${cell.stepCount}. The write changed values, not the state shape.`);
  });
  resetBtn.addEventListener('click', () => reset('Reset complete'));

  scaleVal.textContent = stateScale.value;
  gammaVal.textContent = (+gamma.value).toFixed(2);
  speedVal.textContent = speed.value;
  drawEmptyState();
  announce('Initialized with an all-zero state.');
  schedule();

  return {
    setRunning(next: boolean): void {
      if (running === next) return;
      running = next;
      pauseBtn.textContent = running ? 'Pause' : 'Resume';
      schedule();
    },
  };
}
