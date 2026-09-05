/**
 * Educational BDH-GPU mechanism cell.
 *
 * Equations follow the BDH-GPU formulation in "The Dragon Hatchling: The Missing
 * Link between the Transformer and Models of the Brain" (Kosowski et al., 2025,
 * arXiv:2509.26507, Sec. 3.2) and Pathway's derivation blog:
 *
 *   v*_t = u_t E                           (low-rank value message)
 *   x_t  = scale*(u_t + (v*_t D_x)^+)      (non-negative toy neuron activity)
 *   y_t  = (x_t ρ_{t-1} D_y)^+ ⊙ x_t       (strict-causal read)
 *   ρ_t  = γ·ρ_{t-1} + x_t^T v*_t          (outer-product write)
 *
 * Deviations from the full paper (disclosed): single layer, single head, no layer
 * norm, no multi-head factorization, random untrained E/D_x/D_y, token input u_t
 * injected directly (in the full model it arrives from the previous layer), and
 * optional scalar decay γ used as a transparent forgetting intervention rather
 * than a claim about the paper's full positional operator U. In the real model
 * x carries a same-token, previous-layer residual. This cell isolates the
 * recurrent memory mechanism; it is NOT an official or trained Pathway model.
 */

import { CausalLinearAttentionState } from './attention.js';

export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export interface BDHConfig {
  n: number;      // number of neurons (N)
  d: number;      // low-rank dimension (D), D << N
  scale: number;  // state update scale
  gamma: number;  // educational retention gate; not the paper's full positional U
  seed: number;
}

export interface StepResult {
  x: Float32Array;        // toy neuron activity (N), non-negative
  y: Float32Array;        // non-negative output (N)
  v: Float32Array;        // low-rank message (D)
  hebbianWrite: Float32Array; // x_t ⊗ v*_t (N*D), the raw write this step made
  stateRead: Float32Array; // x_t^T σ_{t-1} (D), before the current write
  activeCount: number;    // number of neurons with |x| above a small epsilon
  /** @deprecated A fixed-state cell cannot expose per-token weights without history. */
  retrievalWeights: Float32Array;
  tokenIndex: number;
}

export class BDHCell {
  readonly cfg: BDHConfig;
  readonly E: Float32Array;    // N x D  (read: y -> latent)
  readonly Dx: Float32Array;   // D x N  (write: latent -> neurons)
  readonly Dy: Float32Array;   // D x N  (read: latent -> neurons)
  readonly embeddings: Map<number, Float32Array>; // token id -> u (N)

  private x: Float32Array;
  private y: Float32Array;
  readonly sigma: Float32Array; // N x D compressed state ρ (legacy public name)
  private readonly memory: CausalLinearAttentionState;
  private t = 0;

  constructor(cfg: BDHConfig) {
    this.cfg = cfg;
    const { n, d, seed } = cfg;
    const rnd = mulberry32(seed);
    const gauss = () => {
      // Box-Muller
      const u1 = Math.max(rnd(), 1e-9), u2 = rnd();
      return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2) * 0.02;
    };
    this.E = new Float32Array(n * d).map(gauss);
    this.Dx = new Float32Array(d * n).map(gauss);
    this.Dy = new Float32Array(d * n).map(gauss);
    this.x = new Float32Array(n);
    this.y = new Float32Array(n);
    this.memory = new CausalLinearAttentionState(n, d, cfg.gamma);
    this.sigma = this.memory.state;
    this.embeddings = new Map();
  }

  /**
   * Sparse unit-norm embedding for a token id (deterministic). Only n/8 random
   * positions are active, with positive values — an untrained stand-in for the
   * sparse, part-whole-style representations BDH's ReLU thresholding produces.
   */
  embedding(token: number): Float32Array {
    let u = this.embeddings.get(token);
    if (!u) {
      const rnd = mulberry32(this.cfg.seed * 7919 + token * 104729);
      u = new Float32Array(this.cfg.n);
      const k = Math.max(4, Math.floor(this.cfg.n / 8));
      const idx = new Set<number>();
      while (idx.size < k) idx.add(Math.floor(rnd() * this.cfg.n));
      let norm = 0;
      for (const i of idx) {
        const v = 0.5 + rnd();
        u[i] = v;
        norm += v * v;
      }
      norm = Math.sqrt(norm);
      for (const i of idx) u[i] /= norm;
      this.embeddings.set(token, u);
    }
    return u;
  }

  reset(): void {
    this.x.fill(0); this.y.fill(0); this.memory.reset();
    this.t = 0;
  }

  get stepCount(): number { return this.t; }

  /** Effective synapse matrix S = σ D_y (N x N, rank D) — what the "graph of synapses" looks like now. */
  synapseMatrix(): Float32Array {
    const { n, d } = this.cfg;
    const S = new Float32Array(n * n);
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < d; k++) {
        const s = this.sigma[i * d + k];
        if (s === 0) continue;
        for (let j = 0; j < n; j++) S[i * n + j] += s * this.Dy[k * n + j];
      }
    }
    return S;
  }

  step(token: number): StepResult {
    const { n, d, scale } = this.cfg;
    const u = this.embedding(token);

    // v*_t = u_t E (R^N -> R^D). In BDH-GPU this input is y_{t,l-1}, the
    // previous layer at the same token. The single-layer toy uses u_t directly;
    // it intentionally adds no extra recurrence across token time.
    const v = new Float32Array(d);
    for (let k = 0; k < d; k++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += u[i] * this.E[i * d + k];
      v[k] = s;
    }

    // x_t = scale*(u_t + (v*_t D_x)^+)  — the token enters fresh each step;
    // unlike our earlier draft there is NO cross-token residual in x: all
    // sequence memory lives in σ, as in the real per-token layer computation.
    const topDown = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let k = 0; k < d; k++) s += v[k] * this.Dx[k * n + i];
      topDown[i] = Math.max(0, s); // (·)^+
    }
    for (let i = 0; i < n; i++) this.x[i] = scale * (u[i] + topDown[i]);

    // y_t = (x_t σ_{t-1} D_y)^+ ⊙ x_t   (readout through the pre-write state)
    const yNew = new Float32Array(n);
    const readout = this.memory.read(this.x); // x_t^T σ_{t-1}; current token excluded
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let k = 0; k < d; k++) s += readout[k] * this.Dy[k * n + j];
      yNew[j] = Math.max(0, s) * this.x[j];
    }

    // Only after producing y_t do we advance the memory. `write` updates every
    // state entry, so even rows where x_t[i] is zero receive the decay factor.
    const write = this.memory.write(this.x, v);
    this.y = yNew;
    this.t++;

    let activeCount = 0;
    for (let i = 0; i < n; i++) if (Math.abs(this.x[i]) > 1e-4) activeCount++;

    return {
      x: Float32Array.from(this.x),
      y: Float32Array.from(this.y),
      v: Float32Array.from(v),
      hebbianWrite: write,
      stateRead: readout,
      activeCount,
      // Per-token decomposition requires O(t) explicit history. It lives in
      // ExplicitHistoryAttention, not in this fixed-shape cell.
      retrievalWeights: new Float32Array(0),
      tokenIndex: this.t - 1,
    };
  }
}

/** Fraction of non-negative activations above zero (sparsity readout, cf. ~5% in the paper). */
export function activeFraction(x: Float32Array, eps = 1e-4): number {
  let a = 0;
  for (let i = 0; i < x.length; i++) if (Math.abs(x[i]) > eps) a++;
  return a / x.length;
}
