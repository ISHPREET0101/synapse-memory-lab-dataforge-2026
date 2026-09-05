/**
 * Linear associative memory capacity experiment (Lab 3).
 *
 * This is the minimal abstraction of the BDH synaptic write: store associations
 * k_i -> v_i with Hebbian outer-product writes into a fixed-size state
 *   σ = Σ_i k_i ⊗ v_i
 * then cue with k_i and read out v̂ = σ k_i. Retrieval degrades as the number of
 * stored associations approaches the state dimension d — interference is the
 * price of a fixed-size memory. This is the same phenomenon the PS asks learners
 * to see ("a fixed-shape recurrent state ... can still forget through
 * interference").
 */

export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export interface CapacityPoint {
  load: number;          // number of associations stored
  recall: number;        // nearest-value retrieval accuracy
  meanCosine: number;    // mean cosine similarity of readout vs true value
  meanCosineOther: number; // legacy alias: mean maximum wrong-value cosine
  maxWrongCosine: number;  // mean strongest incorrect match (interference competitor)
}

function randUnit(rnd: () => number, d: number): Float32Array {
  const v = new Float32Array(d);
  let norm = 0;
  for (let i = 0; i < d; i++) { v[i] = rnd() * 2 - 1; norm += v[i] * v[i]; }
  norm = Math.sqrt(norm);
  for (let i = 0; i < d; i++) v[i] /= norm;
  return v;
}

export function runCapacityTrial(d: number, load: number, seed = 42): CapacityPoint {
  if (!Number.isInteger(d) || d <= 0) throw new RangeError('d must be a positive integer');
  if (!Number.isInteger(load) || load < 0) throw new RangeError('load must be a non-negative integer');
  if (!Number.isFinite(seed)) throw new RangeError('seed must be finite');
  const rnd = mulberry32(seed * 100003 + load);
  const keys: Float32Array[] = [], vals: Float32Array[] = [];
  const sigma = new Float32Array(d * d);
  for (let i = 0; i < load; i++) {
    const k = randUnit(rnd, d), v = randUnit(rnd, d);
    keys.push(k); vals.push(v);
    for (let a = 0; a < d; a++) {
      const ka = k[a];
      if (ka === 0) continue;
      for (let b = 0; b < d; b++) sigma[a * d + b] += ka * v[b];
    }
  }
  let correct = 0, cosSum = 0, maxWrongSum = 0;
  for (let i = 0; i < load; i++) {
    const vhat = new Float32Array(d);
    const k = keys[i];
    for (let b = 0; b < d; b++) {
      let s = 0;
      for (let a = 0; a < d; a++) s += sigma[a * d + b] * k[a];
      vhat[b] = s;
    }
    const cos = (p: Float32Array, q: Float32Array) => {
      let dot = 0, np = 0, nq = 0;
      for (let j = 0; j < d; j++) { dot += p[j] * q[j]; np += p[j] * p[j]; nq += q[j] * q[j]; }
      return dot / (Math.sqrt(np) * Math.sqrt(nq) + 1e-12);
    };
    const trueCosine = cos(vhat, vals[i]);
    cosSum += trueCosine;
    let bestIndex = 0;
    let bestCosine = -Infinity;
    let maxWrong = -Infinity;
    for (let candidate = 0; candidate < load; candidate++) {
      const score = cos(vhat, vals[candidate]);
      if (score > bestCosine) { bestCosine = score; bestIndex = candidate; }
      if (candidate !== i && score > maxWrong) maxWrong = score;
    }
    if (bestIndex === i) correct++;
    maxWrongSum += Number.isFinite(maxWrong) ? maxWrong : 0;
  }
  return {
    load,
    recall: load === 0 ? 1 : correct / load,
    meanCosine: load === 0 ? 1 : cosSum / load,
    meanCosineOther: load ? maxWrongSum / load : 0,
    maxWrongCosine: load ? maxWrongSum / load : 0,
  };
}

export function runCapacityExperiment(d: number, loads: number[], seed = 42): CapacityPoint[] {
  if (!Number.isInteger(d) || d <= 0) throw new RangeError('d must be a positive integer');
  if (!Array.isArray(loads)) throw new TypeError('loads must be an array');
  // average 3 seeds per point so the live curve is smooth but still honest
  return loads.map((L) => {
    const runs = [0, 1, 2].map((i) => runCapacityTrial(d, L, seed + i * 17));
    const avg = (f: (p: CapacityPoint) => number) => runs.reduce((a, p) => a + f(p), 0) / runs.length;
    return {
      load: L,
      recall: avg((p) => p.recall),
      meanCosine: avg((p) => p.meanCosine),
      meanCosineOther: avg((p) => p.meanCosineOther),
      maxWrongCosine: avg((p) => p.maxWrongCosine),
    };
  });
}
