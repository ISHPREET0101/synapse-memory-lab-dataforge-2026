/**
 * Attention primitives used by the labs.
 *
 * `CausalLinearAttentionState` is the constant-memory implementation: for key
 * k_t and value v_t it maintains only
 *
 *   S_t = decay * S_{t-1} + k_t ⊗ v_t.
 *
 * A query at position t reads q_t^T S_{t-1}, before k_t/v_t are written. This
 * strict ordering makes the operation causal and excludes the current token.
 * `ExplicitHistoryAttention` is deliberately separate. It retains every key
 * and value and exists only as a transparent reference implementation for
 * tests and educational comparisons; it must not be embedded in a fixed-state
 * model cell.
 */

export interface LinearAttentionStep {
  /** q_t^T S_{t-1}; the current key/value is not included. */
  read: Float32Array;
  /** k_t ⊗ v_t, before decay and accumulation. */
  write: Float32Array;
}

function assertDimension(name: string, vector: ArrayLike<number>, expected: number): void {
  if (vector.length !== expected) {
    throw new RangeError(`${name} has length ${vector.length}; expected ${expected}`);
  }
}

function assertDecay(decay: number): void {
  if (!Number.isFinite(decay) || decay < 0 || decay > 1) {
    throw new RangeError(`decay must be finite and in [0, 1]; received ${decay}`);
  }
}

/** Fixed-shape, strict-causal linear-attention/Hebbian accumulator. */
export class CausalLinearAttentionState {
  readonly keyDimension: number;
  readonly valueDimension: number;
  readonly decay: number;
  /** Row-major keyDimension × valueDimension matrix. Its size never changes. */
  readonly state: Float32Array;

  constructor(keyDimension: number, valueDimension: number, decay: number) {
    if (!Number.isInteger(keyDimension) || keyDimension <= 0) {
      throw new RangeError('keyDimension must be a positive integer');
    }
    if (!Number.isInteger(valueDimension) || valueDimension <= 0) {
      throw new RangeError('valueDimension must be a positive integer');
    }
    assertDecay(decay);
    this.keyDimension = keyDimension;
    this.valueDimension = valueDimension;
    this.decay = decay;
    this.state = new Float32Array(keyDimension * valueDimension);
  }

  reset(): void {
    this.state.fill(0);
  }

  /** Read the state as it exists before the current token is written. */
  read(query: ArrayLike<number>): Float32Array {
    assertDimension('query', query, this.keyDimension);
    const out = new Float32Array(this.valueDimension);
    for (let j = 0; j < this.valueDimension; j++) {
      let sum = 0;
      for (let i = 0; i < this.keyDimension; i++) {
        sum += query[i] * this.state[i * this.valueDimension + j];
      }
      out[j] = sum;
    }
    return out;
  }

  /**
   * Apply S <- decay*S + key⊗value. Every entry is updated, including rows
   * whose current key component is zero, so old state always decays correctly.
   */
  write(key: ArrayLike<number>, value: ArrayLike<number>): Float32Array {
    assertDimension('key', key, this.keyDimension);
    assertDimension('value', value, this.valueDimension);
    const write = new Float32Array(this.state.length);
    for (let i = 0; i < this.keyDimension; i++) {
      for (let j = 0; j < this.valueDimension; j++) {
        const index = i * this.valueDimension + j;
        const contribution = key[i] * value[j];
        write[index] = contribution;
        this.state[index] = this.decay * this.state[index] + contribution;
      }
    }
    return write;
  }

  /** Strict-causal step: read S_{t-1}, then write k_t⊗v_t into S_t. */
  step(
    query: ArrayLike<number>,
    key: ArrayLike<number>,
    value: ArrayLike<number>,
  ): LinearAttentionStep {
    const read = this.read(query);
    const write = this.write(key, value);
    return { read, write };
  }
}

export interface AttentionHistoryEntry {
  key: Float32Array;
  value: Float32Array;
}

/**
 * Direct O(t) history sum corresponding to q_t^T S_{t-1}:
 * sum_{τ<t} decay^(t-1-τ) (q_t·k_τ) v_τ.
 */
export function explicitHistoryRead(
  query: ArrayLike<number>,
  history: ReadonlyArray<AttentionHistoryEntry>,
  decay: number,
): Float32Array {
  assertDecay(decay);
  if (history.length === 0) return new Float32Array(0);
  const keyDimension = history[0].key.length;
  const valueDimension = history[0].value.length;
  assertDimension('query', query, keyDimension);
  const out = new Float32Array(valueDimension);
  for (let tau = 0; tau < history.length; tau++) {
    const { key, value } = history[tau];
    assertDimension(`history[${tau}].key`, key, keyDimension);
    assertDimension(`history[${tau}].value`, value, valueDimension);
    let similarity = 0;
    for (let i = 0; i < keyDimension; i++) similarity += query[i] * key[i];
    const weight = Math.pow(decay, history.length - 1 - tau) * similarity;
    for (let j = 0; j < valueDimension; j++) out[j] += weight * value[j];
  }
  return out;
}

/** Growing-history oracle. Use for validation/visualization, never as the cell state. */
export class ExplicitHistoryAttention {
  readonly decay: number;
  private readonly entries: AttentionHistoryEntry[] = [];

  constructor(decay: number) {
    assertDecay(decay);
    this.decay = decay;
  }

  get length(): number {
    return this.entries.length;
  }

  reset(): void {
    this.entries.length = 0;
  }

  read(query: ArrayLike<number>): Float32Array {
    return explicitHistoryRead(query, this.entries, this.decay);
  }

  step(
    query: ArrayLike<number>,
    key: ArrayLike<number>,
    value: ArrayLike<number>,
  ): Float32Array {
    if (key.length === 0 || value.length === 0) {
      throw new RangeError('key and value must be non-empty');
    }
    if (this.entries.length > 0) {
      assertDimension('key', key, this.entries[0].key.length);
      assertDimension('value', value, this.entries[0].value.length);
      assertDimension('query', query, this.entries[0].key.length);
    } else {
      assertDimension('query', query, key.length);
    }
    const read = this.entries.length === 0
      ? new Float32Array(value.length)
      : this.read(query);
    this.entries.push({ key: Float32Array.from(key), value: Float32Array.from(value) });
    return read;
  }
}

/**
 * Softmax baseline using the same per-token vectors as the toy cell. Keys and
 * queries are x_τ; only positions τ < t are visible.
 */

export function softmaxAttentionWeights(
  xs: Float32Array[],
  t: number,
): Float32Array {
  if (xs.length === 0) throw new RangeError('xs must contain at least one vector');
  if (!Number.isInteger(t) || t < 0 || t >= xs.length) {
    throw new RangeError(`t must index xs; received ${t} for length ${xs.length}`);
  }
  if (t === 0) return new Float32Array(0);
  const n = xs[0].length;
  if (n === 0) throw new RangeError('attention vectors must be non-empty');
  for (let i = 1; i < xs.length; i++) {
    assertDimension(`xs[${i}]`, xs[i], n);
  }
  const scores = new Float32Array(t); // scores[τ] for τ < t
  let max = -Infinity;
  for (let tau = 0; tau < t; tau++) {
    let dot = 0;
    const xt = xs[tau];
    for (let i = 0; i < n; i++) dot += xs[t][i] * xt[i];
    const s = dot / Math.sqrt(n);
    scores[tau] = s;
    if (s > max) max = s;
  }
  let sum = 0;
  for (let tau = 0; tau < t; tau++) {
    scores[tau] = Math.exp(scores[tau] - max);
    sum += scores[tau];
  }
  for (let tau = 0; tau < t; tau++) scores[tau] /= sum;
  return scores;
}

/** Spearman rank correlation: Pearson correlation of average ranks. */
export function spearman(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) throw new RangeError('rank vectors must have equal length');
  const n = a.length;
  if (n < 2) return NaN;
  const rank = (v: Float32Array): Float32Array => {
    const idx = Array.from({ length: n }, (_, i) => i).sort((i, j) => v[i] - v[j]);
    const r = new Float32Array(n);
    let i = 0;
    while (i < n) {
      let j = i;
      while (j + 1 < n && v[idx[j + 1]] === v[idx[i]]) j++;
      const avg = (i + j) / 2 + 1;
      for (let k = i; k <= j; k++) r[idx[k]] = avg;
      i = j + 1;
    }
    return r;
  };
  const ra = rank(a), rb = rank(b);
  let meanA = 0, meanB = 0;
  for (let i = 0; i < n; i++) { meanA += ra[i]; meanB += rb[i]; }
  meanA /= n; meanB /= n;
  let covariance = 0, varianceA = 0, varianceB = 0;
  for (let i = 0; i < n; i++) {
    const da = ra[i] - meanA, db = rb[i] - meanB;
    covariance += da * db;
    varianceA += da * da;
    varianceB += db * db;
  }
  const denominator = Math.sqrt(varianceA * varianceB);
  return denominator === 0 ? NaN : covariance / denominator;
}
