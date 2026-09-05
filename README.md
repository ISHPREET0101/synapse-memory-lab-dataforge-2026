# Synapse Memory Lab

Synapse Memory Lab is a self-contained interactive visual essay for the DataForge 2026 Pathway Track. It teaches one falsifiable claim:

> BDH-GPU's causal linear attention can be evaluated through recurrent Hebbian-like outer-product writes to a fixed-shape synaptic state; this removes sequence-length growth from the recurrent state, while finite state can still attenuate or mix old associations.

The browser experience opens with a deterministic toy simulation already running. Learners can change the update scale, decay, query position, replay speed, associative-memory dimension, and load, then see the consequences immediately.

## Submission status

- **Local artifact:** implemented, tested, and buildable.
- **Private GitHub review repository:** https://github.com/ISHPREET0101/synapse-memory-lab-dataforge-2026
- **Public artifact URL:** **not deployed yet**.
- **Public source repository:** **not published yet**.
- **Official BDH/BDH-CQ checkpoint:** **not run**. No live checkpoint is bundled or called.
- **Evidence boundary:** the labs are local toy/abstracted computations; paper metrics and architecture properties are author-reported, not independently reproduced here.

The private repository is available for team review; PS-1 still requires a public source repository and a no-sign-in public artifact. Do not replace the status above with placeholders presented as live links.

## Audience, prerequisites, and learning objectives

**Audience:** ML-literate students and practitioners who know Transformer attention but have not studied BDH.

**Prerequisites:** vectors and matrix multiplication, dot-product/softmax attention, causal sequence processing, and the purpose of a Transformer KV cache. No neuroscience background is needed.

After the guided tour, a learner should be able to:

1. derive an outer-product memory update and its query readout;
2. distinguish a fixed-shape recurrent state from a history-growing KV cache;
3. identify activity `x`, low-rank message `v*`, compressed state `rho` (the code's legacy `sigma` field), and frozen projections in the toy BDH cell;
4. explain why read-before-write ordering is causal;
5. predict how decay and memory load affect retrieval;
6. distinguish the live toy, the abstract capacity experiment, and author-reported BDH/BDH-CQ evidence;
7. name the toy's deviations and avoid calling it an official Pathway model or a calibrated BDH benchmark.

## What is implemented

### Lab 1 — watch memory being written

A fixed token stream drives a 96-neuron, rank-12 `BDHCell`. The left canvas renders the current neuron vector. The right canvas renders the effective `96 x 96` matrix `sigma Dy`, whose rank is at most 12. Gold rings indicate neurons contributing to the current outer-product write. Controls rebuild or advance the deterministic cell.

The cell keeps `sigma` with shape `N x D`; its allocation does not grow with token count. The preset tokens are repeated integer IDs with deterministic sparse embeddings—not natural-language model tokens or learned embeddings.

### Lab 2 — compare attention profiles

For a chosen causal query position, the lab evaluates the same decayed linear-attention operation in two ways: once by reading a fixed `12 x 8` recurrent state, and once by summing the prior key/value history explicitly. It displays both output vectors and their maximum absolute error. A softmax output over the same causal history appears as a separately labeled contrast; it is not claimed to be equivalent.

The explicit-history oracle is deliberately diagnostic. It proves the recurrent reordering for this implemented linear operation without pretending that a fixed aggregate state can expose a token-by-token attention map for free.

### Lab 3 — provoke interference

A separate linear associative-memory experiment draws deterministic random unit key/value pairs, writes `sigma = sum_i k_i outer v_i`, and retrieves with a key. The curve averages three seeded trials at each load and reports:

- recall: nearest-stored-value retrieval accuracy;
- mean cosine to the true value; and
- mean strongest wrong-value cosine as an interference reference.

This is an exact experiment for the implemented linear associative memory, not a measured capacity curve for trained BDH.

### BDH module and guided tour

The page connects the manipulated variables to the BDH-GPU equations and summarizes BDH-CQ as reported by its authors. A five-step overlay guides the learner from claim to failure mode. The comparison table is a hand-authored synthesis of cited source properties, not model output.

## Architecture

```text
index.html (content, controls, paper-backed comparison)
    |
    +-- src/main.ts (initialization and lifecycle)
          |
          +-- src/labs/lab1.ts --> src/engine/bdh.ts
          |                         +-- src/engine/attention.ts
          |                         +-- neuron/synapse canvas renderers
          |
          +-- src/labs/lab2.ts --> recurrent state + explicit-history oracle
          |
          +-- src/labs/lab3.ts --> src/engine/retrieval.ts
          |                         +-- line-chart renderer
          |
          +-- src/content/tour.ts

Vite + TypeScript --> dist/ (static deployment bundle)
Vitest --> tests/engine.test.ts
```

### Component map

| Component | Role |
|---|---|
| `index.html` | Semantic page shell, teaching narrative, controls, equations, evidence labels, comparison table, and source links. |
| `src/main.ts` | Starts all labs and the tour; pauses Lab 1 when hidden or off-screen. |
| `src/engine/bdh.ts` | Deterministic, untrained, single-layer toy BDH-GPU-style cell with fixed `N x D` synaptic state. |
| `src/engine/attention.ts` | Strict-causal fixed-state outer-product accumulator, explicit-history oracle, softmax baseline, and Spearman statistic. |
| `src/engine/retrieval.ts` | Seeded linear associative-memory capacity/interference trials. |
| `src/labs/lab1.ts` | Streaming preset, controls, cell lifecycle, and neuron/write readouts. |
| `src/labs/lab2.ts` | Numerical equality check between recurrent and explicit-history linear attention, plus a labeled softmax contrast. |
| `src/labs/lab3.ts` | Live capacity curve and single cue stress test. |
| `src/viz/*.ts` | Dependency-free Canvas 2D neuron, matrix, and chart renderers. |
| `src/content/tour.ts` | Five-step guide and keyboard controls. |
| `src/styles/main.css` | Responsive visual system and evidence tags. |
| `tests/engine.test.ts` | Unit/contract checks for causality, fixed state, determinism, attention, decay, dimensions, and capacity behavior. |
| `ARCHITECTURE.md` | Current implementation map, invariants, equations, and evidence boundaries. |
| `dist/` | Generated production output. Rebuild it; do not hand-edit it. |

## Evidence ledger: live, synthetic, precomputed

| Element | Status | Meaning |
|---|---|---|
| Lab 1 state and canvases | **Live + synthetic** | Computed in the browser from preset integer tokens, seeded embeddings, and random untrained matrices. |
| Lab 2 vector bars and max error | **Live + synthetic** | Recomputed from the toy sequence on each control change; equality is checked against an explicit-history oracle. |
| Lab 3 curve and cue result | **Live + synthetic** | Deterministic seeded random associations; curve is calculated in-browser, not loaded from a file. |
| BDH-vs-Transformer table | **Precomputed synthesis** | Static prose/table distilled from primary sources; no benchmark is run. |
| Approximately 5% activity, scaling statements, BDH-CQ ARC/cost numbers | **Author-reported** | Cited paper claims shown for context, not reproduced or verified by this project. |
| Animation | **Computed rendering** | Canvas frames visualize current computed arrays. There is no prerecorded or scripted fake model trace. |
| Data, model weights, network APIs | **None** | No external dataset, learned checkpoint, backend, analytics, cookies, or post-load network request is required. |

## Setup and reproduction

Prerequisite: Node.js with npm. The current build was verified with Node `v24.19.0` and npm `11.17.0`; the lockfile is the reproducibility authority.

```bash
npm ci
npm run dev
```

Open the local URL printed by Vite (normally `http://localhost:5173`). No environment variables are required; `.env.example` documents that fact.

To reproduce the review build:

```bash
npm ci
npm test
npm run build
npm run preview
```

Expected local verification at the time of this README:

- Vitest: `19` tests passed in `tests/engine.test.ts`.
- Production build: TypeScript strict check passed and Vite emitted the static `dist/` bundle.

These are local software checks. They are not leaderboard results, research replication, or evidence that an official BDH checkpoint was executed.

### Sixty-second claim test

1. In Lab 1, set `gamma` near `1`, single-step repeated words, and watch `sigma Dy` accumulate.
2. Set `gamma` to `0`; old state stops accumulating and is replaced by the newest write.
3. In Lab 2, move query and decay; confirm the recurrent and explicit-history outputs stay numerically aligned while softmax can differ.
4. In Lab 3, increase load beyond dimension `d`; watch nearest-value accuracy degrade as interference grows.

## Tests

`npm test` checks:

- deterministic cell output under the same seed;
- non-negative toy output;
- fixed state size after 500 steps;
- reset and decay behavior;
- repeated-token similarity behavior;
- strict-causal read-before-write semantics;
- agreement between fixed-state and explicit-history linear attention;
- dimension/input validation;
- softmax normalization and Spearman ranking; and
- degradation of the implemented capacity experiment beyond low load.

The tests do **not** establish equation-level equivalence to all details of the full BDH paper, model quality, GPU performance, learned sparsity, ARC performance, or deployment health.

## Limitations and known interpretation traps

- This is an independent toy reimplementation, not official Pathway code or a trained BDH model.
- It is one layer and one head, with no layer normalization or multi-head factorization. `E`, `Dx`, and `Dy` are seeded random matrices.
- Integer token IDs map to synthetic, sparse, positive, unit-norm embeddings. There is no tokenizer or language training.
- The paper's positional operator is simplified to scalar decay `gamma I`; rotation-block behavior is not implemented.
- The toy uses a hand-set scale and injects token input directly. Quantitative behavior must not be extrapolated to trained BDH.
- In this code, `x` and `y` are non-negative, but `v*`, the raw `x outer v*` write, and effective `sigma Dy` can be signed because the random projection matrices are signed. Therefore the page's broad “non-negative update” wording should be understood as non-negative neural activity/readout in the toy, not a guarantee that every stored matrix entry is non-negative.
- Lab 2's explicit oracle retains history only to validate the same linear result. A fixed aggregate state alone does not preserve a free, exact per-token attention map.
- Lab 2 proves equality only for the implemented causal decayed linear-attention operation; the softmax output is a non-equivalent contrast.
- Lab 3 implements a generic linear associative memory. Nearest-value accuracy and strongest-wrong-value similarity are toy diagnostics, not BDH benchmarks.
- `O(1)` means constant in **sequence length** for a fixed architecture. The state still costs `O(ND)` per head/layer.
- Fixed state supports arbitrarily many update steps, not perfect recall of arbitrarily many items.
- Claims about trained sparsity, heavy-tailed connectivity, GPT-2 comparisons, BDH-CQ results, cost, and latent reasoning are author-reported citations. They were not independently reproduced.
- No live BDH/BDH-CQ checkpoint has been run. No public URL or public repository is currently deployed.

## AI assistance disclosure

OpenAI Codex was used as an assistant during drafting and editing of code, explanatory prose, research organization, tests, and documentation. AI output was not treated as experimental evidence. The submitting team is responsible for checking the implementation, sources, licenses, and every claim, and must be able to derive and defend the system without relying on the assistant. Any later human reviewers, mentors, generators, or reused assets should be appended to this disclosure before submission.

## Credits and licenses

The original project code, prose, and Canvas graphics are released under the MIT License in [`LICENSE`](LICENSE). The BDH equations and reported results are credited to their authors; citation does not relicense the papers. The Pathway reference repository is MIT-licensed, but this project is described as an independent implementation, not an official distribution.

See:

- [`docs/PAPERS.md`](docs/PAPERS.md) for the claim-to-source map;
- [`docs/SOURCE_AND_LICENSES.md`](docs/SOURCE_AND_LICENSES.md) for code, data, model, asset, font, and dependency provenance; and
- [`docs/JUDGE_DEFENSE.md`](docs/JUDGE_DEFENSE.md) for derivations, implementation tracing, and anticipated questions.

## Deployment

The app is a static Vite build. Run `npm run build`, then publish the contents of `dist/` to a static host. `vite.config.ts` uses `base: './'`, so relative assets work from a project subpath (including GitHub Pages).

Before declaring the submission deployed:

1. publish the source repository publicly;
2. configure the host to deploy `dist/` (or run `npm ci && npm run build` and use `dist` as the output directory);
3. open the public artifact in a signed-out/private browser window;
4. confirm all three labs move and controls recompute;
5. confirm the source links and mobile layout;
6. rerun `npm test` and `npm run build` from a clean checkout; and
7. replace this README's status with the verified public artifact and repository URLs.

At present, deployment instructions are ready, but **there is no verified public deployment or public repository URL**.
