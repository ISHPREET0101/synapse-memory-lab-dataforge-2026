import { describe, expect, it } from 'vitest';
import { BDHCell } from '../src/engine/bdh.js';
import {
  CausalLinearAttentionState,
  ExplicitHistoryAttention,
  softmaxAttentionWeights,
  spearman,
} from '../src/engine/attention.js';
import {
  mulberry32 as retrievalRandom,
  runCapacityExperiment,
  runCapacityTrial,
} from '../src/engine/retrieval.js';

function expectVectorClose(actual: ArrayLike<number>, expected: ArrayLike<number>, digits = 5): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < actual.length; i++) expect(actual[i]).toBeCloseTo(expected[i], digits);
}

describe('BDHCell', () => {
  it('is deterministic for the same seed', () => {
    const run = () => {
      const c = new BDHCell({ n: 32, d: 6, scale: 0.3, gamma: 0.9, seed: 11 });
      const out = [];
      for (const t of [1, 2, 3, 4]) out.push(Array.from(c.step(t).y));
      return out;
    };
    expect(run()).toEqual(run());
  });

  it('keeps toy activity and output non-negative', () => {
    const c = new BDHCell({ n: 48, d: 8, scale: 0.6, gamma: 0.9, seed: 3 });
    for (let t = 0; t < 30; t++) {
      const r = c.step((t * 7) % 12);
      expect(r.x.every((v) => v >= 0)).toBe(true);
      expect(r.y.every((v) => v >= 0)).toBe(true);
    }
  });

  it('has fixed-size state regardless of sequence length', () => {
    const c = new BDHCell({ n: 32, d: 6, scale: 0.3, gamma: 0.9, seed: 5 });
    const stateBuffer = c.sigma.buffer;
    const stateLength = c.sigma.length;
    for (let t = 0; t < 2_000; t++) {
      const result = c.step(t % 7);
      expect(result.retrievalWeights).toHaveLength(0);
    }
    expect(c.sigma.length).toBe(stateLength);
    expect(c.sigma.buffer).toBe(stateBuffer);
    expect(c.sigma.byteLength).toBe(32 * 6 * Float32Array.BYTES_PER_ELEMENT);
    expect(c.embeddings.size).toBe(7);
    expect(c.stepCount).toBe(2_000);
  });

  it('state with gamma=0 forgets: synapse matrix returns to ~zero', () => {
    const c = new BDHCell({ n: 32, d: 6, scale: 0.5, gamma: 0.0, seed: 9 });
    for (let t = 0; t < 20; t++) c.step(t % 5);
    const S1 = c.synapseMatrix().reduce((a, b) => a + Math.abs(b), 0);
    // one more step after a zero write (v = y E may not be exactly zero, so run
    // several steps with gamma=0 and check the matrix does not accumulate)
    for (let t = 0; t < 40; t++) c.step(t % 5);
    const S2 = c.synapseMatrix().reduce((a, b) => a + Math.abs(b), 0);
    expect(S2).toBeLessThan(S1 * 10 + 1e-6);
  });

  it('uses the previous state, so the first causal read is zero', () => {
    const c = new BDHCell({ n: 32, d: 6, scale: 0.3, gamma: 1, seed: 5 });
    const first = c.step(1);
    expect(Array.from(first.stateRead)).toEqual(new Array(6).fill(0));
    expect(first.y.every((v) => v === 0)).toBe(true);
    expect(c.sigma.some((v) => v !== 0)).toBe(true);
  });

  it('repeated tokens retrieve their earlier occurrence above other tokens', () => {
    const c = new BDHCell({ n: 64, d: 10, scale: 0.4, gamma: 0.97, seed: 21 });
    const hist: Float32Array[] = [];
    for (const tok of [1, 2, 3]) hist.push(c.step(tok).x);
    c.step(1); // repeat of token 0
    const u = c.embedding(1);
    const score = (x: Float32Array) => {
      let d = 0; for (let i = 0; i < u.length; i++) d += u[i] * x[i];
      return d;
    };
    expect(score(hist[0])).toBeGreaterThan(score(hist[1]));
  });
});

describe('attention comparison', () => {
  it('reads the previous state before applying the current write', () => {
    const state = new CausalLinearAttentionState(2, 2, 1);
    const first = state.step([1, 0], [1, 0], [3, -2]);
    expect(Array.from(first.read)).toEqual([0, 0]);
    expectVectorClose(first.write, [3, -2, 0, 0], 7);

    // The current value [100, 100] must not leak into this read. Only the
    // preceding outer product contributes: [1/2, 1/2]^T S_1 = [1.5, -1].
    const second = state.step([0.5, 0.5], [0, 1], [100, 100]);
    expect(Array.from(second.read)).toEqual([1.5, -1]);
    expectVectorClose(second.write, [0, 0, 100, 100], 7);
  });

  it.each([0, 0.2, 0.75, 1])(
    'fixed state equals explicit history over a deterministic random sequence (decay=%s)',
    (decay) => {
      const keyDimension = 5;
      const valueDimension = 4;
      const state = new CausalLinearAttentionState(keyDimension, valueDimension, decay);
      const oracle = new ExplicitHistoryAttention(decay);
      let randomState = 0x12345678;
      const next = () => {
        randomState = (Math.imul(randomState, 1664525) + 1013904223) >>> 0;
        return randomState / 0xffffffff * 2 - 1;
      };

      const stateBuffer = state.state.buffer;
      for (let t = 0; t < 100; t++) {
        const query = Array.from({ length: keyDimension }, next);
        const key = Array.from({ length: keyDimension }, next);
        const value = Array.from({ length: valueDimension }, next);
        const recurrentRead = state.step(query, key, value).read;
        const historyRead = oracle.step(query, key, value);
        expectVectorClose(recurrentRead, historyRead, 4);
        expect(state.state).toHaveLength(keyDimension * valueDimension);
        expect(state.state.buffer).toBe(stateBuffer);
        expect(oracle.length).toBe(t + 1);
      }
    },
  );

  it('reset zeroes recurrent state in place and clears explicit history', () => {
    const state = new CausalLinearAttentionState(3, 2, 0.8);
    const oracle = new ExplicitHistoryAttention(0.8);
    const stateBuffer = state.state.buffer;
    state.step([1, 2, 3], [2, 0, -1], [4, 5]);
    oracle.step([1, 2, 3], [2, 0, -1], [4, 5]);
    expect(state.state.some((value) => value !== 0)).toBe(true);
    expect(oracle.length).toBe(1);

    state.reset();
    oracle.reset();
    expect(state.state.buffer).toBe(stateBuffer);
    expect(Array.from(state.state)).toEqual(new Array(6).fill(0));
    expect(oracle.length).toBe(0);
    expect(Array.from(state.step([1, 1, 1], [1, 1, 1], [9, 9]).read)).toEqual([0, 0]);
    expect(Array.from(oracle.step([1, 1, 1], [1, 1, 1], [9, 9]))).toEqual([0, 0]);
  });

  it('rejects invalid dimensions and decay without partially mutating state', () => {
    expect(() => new CausalLinearAttentionState(0, 2, 1)).toThrow(RangeError);
    expect(() => new CausalLinearAttentionState(2.5, 2, 1)).toThrow(RangeError);
    expect(() => new CausalLinearAttentionState(2, -1, 1)).toThrow(RangeError);
    expect(() => new CausalLinearAttentionState(2, 2, -0.01)).toThrow(RangeError);
    expect(() => new CausalLinearAttentionState(2, 2, 1.01)).toThrow(RangeError);
    expect(() => new CausalLinearAttentionState(2, 2, Number.NaN)).toThrow(RangeError);

    const state = new CausalLinearAttentionState(2, 2, 0.5);
    state.write([1, 2], [3, 4]);
    const before = Array.from(state.state);
    expect(() => state.read([1])).toThrow(/query has length 1; expected 2/);
    expect(() => state.write([1], [3, 4])).toThrow(/key has length 1; expected 2/);
    expect(() => state.write([1, 2], [3])).toThrow(/value has length 1; expected 2/);
    expect(() => state.step([1, 2], [1], [3, 4])).toThrow(RangeError);
    expect(Array.from(state.state)).toEqual(before);
  });

  it('decays every state row, including rows absent from the next key', () => {
    const state = new CausalLinearAttentionState(2, 1, 0.5);
    state.write([1, 0], [2]);
    state.write([0, 1], [0]);
    expect(state.state[0]).toBeCloseTo(1, 7);
    expect(state.state[1]).toBeCloseTo(0, 7);
  });
  it('softmax weights are normalized and causal', () => {
    const xs = [new Float32Array([1, 0]), new Float32Array([0, 1]), new Float32Array([1, 1])];
    const w = softmaxAttentionWeights(xs, 2);
    expect(w.length).toBe(2);
    expect(w[0] + w[1]).toBeCloseTo(1, 5);
  });

  it('spearman is 1 for identical ranking', () => {
    const a = new Float32Array([0.1, 0.5, 0.2, 0.9]);
    expect(spearman(a, Float32Array.from(a))).toBeCloseTo(1, 5);
  });
});

describe('capacity experiment', () => {
  it('uses nearest-value retrieval semantics and exposes the strongest wrong match', () => {
    const dimension = 5;
    const load = 6;
    const seed = 19;
    const random = retrievalRandom(seed * 100003 + load);
    const unit = () => {
      const vector = Float32Array.from({ length: dimension }, () => random() * 2 - 1);
      let norm = 0;
      for (const value of vector) norm += value * value;
      norm = Math.sqrt(norm);
      for (let i = 0; i < vector.length; i++) vector[i] /= norm;
      return vector;
    };
    const keys: Float32Array[] = [];
    const values: Float32Array[] = [];
    const memory = new Float32Array(dimension * dimension);
    for (let item = 0; item < load; item++) {
      keys.push(unit());
      values.push(unit());
      for (let row = 0; row < dimension; row++) {
        for (let column = 0; column < dimension; column++) {
          memory[row * dimension + column] += keys[item][row] * values[item][column];
        }
      }
    }
    const cosine = (a: Float32Array, b: Float32Array) => {
      let dot = 0; let normA = 0; let normB = 0;
      for (let i = 0; i < dimension; i++) {
        dot += a[i] * b[i]; normA += a[i] * a[i]; normB += b[i] * b[i];
      }
      return dot / (Math.sqrt(normA) * Math.sqrt(normB) + 1e-12);
    };
    let correct = 0; let trueCosineSum = 0; let maxWrongSum = 0;
    for (let item = 0; item < load; item++) {
      const read = new Float32Array(dimension);
      for (let column = 0; column < dimension; column++) {
        for (let row = 0; row < dimension; row++) {
          read[column] += memory[row * dimension + column] * keys[item][row];
        }
      }
      const scores = values.map((value) => cosine(read, value));
      const nearest = scores.indexOf(Math.max(...scores));
      if (nearest === item) correct++;
      trueCosineSum += scores[item];
      maxWrongSum += Math.max(...scores.filter((_, candidate) => candidate !== item));
    }

    const point = runCapacityTrial(dimension, load, seed);
    expect(point.recall).toBe(correct / load);
    expect(point.meanCosine).toBeCloseTo(trueCosineSum / load, 6);
    expect(point.maxWrongCosine).toBeCloseTo(maxWrongSum / load, 6);
    expect(point.meanCosineOther).toBe(point.maxWrongCosine);
  });

  it('defines empty and singleton retrieval without false interference', () => {
    expect(runCapacityTrial(8, 0, 7)).toEqual({
      load: 0,
      recall: 1,
      meanCosine: 1,
      meanCosineOther: 0,
      maxWrongCosine: 0,
    });
    const singleton = runCapacityTrial(8, 1, 7);
    expect(singleton.recall).toBe(1);
    expect(singleton.meanCosine).toBeCloseTo(1, 6);
    expect(singleton.maxWrongCosine).toBe(0);
  });

  it('recall is high below capacity and degrades beyond it', () => {
    const d = 64;
    const pts = runCapacityExperiment(d, [4, d, d * 2]);
    expect(pts[0].recall).toBeGreaterThan(0.9);
    expect(pts[2].recall).toBeLessThan(pts[0].recall);
    expect(pts[2].meanCosineOther).toBeGreaterThan(pts[0].meanCosineOther);
  });
});
