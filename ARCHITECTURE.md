# Synapse Memory Lab — implemented architecture

**Track:** DataForge 2026 Pathway PS-1  
**Substrate:** static Vite + strict TypeScript; Canvas 2D; zero runtime packages, backends, analytics, cookies, model checkpoints, or post-load API calls.

## Claim and boundary

BDH-GPU's causal linear attention can be evaluated through recurrent Hebbian-like outer-product writes to a fixed-shape state. For fixed architecture dimensions, the recurrent allocation does not grow with sequence length. Finite state is not infinite memory: decay attenuates old terms and superposed associations interfere.

This repository is an independent, untrained educational toy. It does not claim official Pathway implementation status, exact softmax equivalence, biological fidelity, or reproduced BDH/BDH-CQ benchmark results.

## Runtime map

```text
index.html
  └─ src/main.ts
      ├─ Lab 1 ─ src/labs/lab1.ts ─ src/engine/bdh.ts
      │                            └─ src/engine/attention.ts
      │                            └─ src/viz/neuronCanvas.ts
      │                            └─ src/viz/synapseCanvas.ts
      ├─ Lab 2 ─ src/labs/lab2.ts ─ src/engine/attention.ts
      ├─ Lab 3 ─ src/labs/lab3.ts ─ src/engine/retrieval.ts
      │                            └─ src/viz/chart.ts
      └─ guided tour ───────────── src/content/tour.ts
```

## Core equations

The generic strict-causal accumulator reads before it writes:

```text
r_t = q_tᵀ S_(t-1)
S_t = gamma S_(t-1) + k_t outer v_t
```

Expanding the recurrence gives the explicit-history oracle:

```text
r_t = sum_(tau<t) gamma^(t-1-tau) (q_t dot k_tau) v_tau
```

The toy BDH-style cell maps the same mechanism into a compressed state `rho` (named `sigma` in the public code for compatibility):

```text
v*_t  = u_t E
x_t   = scale [u_t + ReLU(v*_t Dx)]
read  = x_tᵀ rho_(t-1)
y_t   = ReLU(read Dy) elementwise-multiply x_t
rho_t = gamma rho_(t-1) + x_t outer v*_t
```

`rho` has shape `N x D`. Lab 1 derives `rho Dy` only for visualization; the `N x N` matrix is not stored as recurrent state. Reads happen before writes, and decay is applied to every state entry.

## Lab contracts

- **Lab 1 — observe writes.** A preset is already running. Update scale and decay alter the actual recurrence; speed changes only the timer. Neuron activity, raw write magnitude, and derived effective synapses are computed live.
- **Lab 2 — prove the reordering.** A fixed `12 x 8` recurrent state and an explicit history sum evaluate the same decayed linear operation. The UI reports maximum absolute error. Causal softmax is a labeled, non-equivalent contrast.
- **Lab 3 — expose interference.** Seeded unit key/value pairs are superposed in one `d x d` associative matrix. Recall is nearest-value accuracy; strongest-wrong cosine exposes the competing match. Results are generic toy behavior, not calibrated BDH capacity.

## Invariants

- The current key/value pair cannot affect its own causal read.
- State identity, shape, and byte length remain constant across sequence steps.
- A write decays the whole old state, including rows where the current key is zero.
- Recurrent and explicit-history linear outputs agree within floating-point tolerance.
- Reset clears fast state without regenerating seeded slow projections.
- Identical seeds and controls reproduce identical results.

These contracts are exercised in `tests/engine.test.ts` and verified by `npm run check`.

## Evidence ownership

Labs 1–3 are live synthetic computations. The comparison prose and reported paper results are static, cited synthesis. No data file or prerecorded trace is presented as a live model run. Primary-source mapping, provenance, limitations, and AI assistance are documented in `docs/` and `README.md`.
