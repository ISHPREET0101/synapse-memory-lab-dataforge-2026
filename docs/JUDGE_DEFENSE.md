# Judge defense: derive it, trace it, challenge it

This is a speaking and whiteboard guide, not a script to memorize. The defense is strongest when the presenter can derive each operation, point to the implementing file, predict a control's effect, and state the evidence boundary before being prompted.

## Thirty-second opening

“Our claim is that causal linear-attention retrieval can be accumulated as Hebbian-like outer-product writes into a fixed-shape synaptic state. Lab 1 shows the state changing, Lab 2 verifies the recurrent result against an explicit-history oracle while keeping softmax visibly separate, and Lab 3 shows the honest cost: interference. Everything interactive is a deterministic in-browser toy or synthetic experiment. We did not run an official BDH or BDH-CQ checkpoint, and the paper results are explicitly author-reported.”

## Derivation 1: history sum into a fixed matrix

Start with causal linear attention at time `t`:

```text
r_t = sum_(tau<t) (q_t dot k_tau) v_tau
```

Use associativity:

```text
r_t = q_t^T [sum_(tau<t) k_tau outer v_tau]
```

Define the bracketed term as state:

```text
S_(t-1) = sum_(tau<t) k_tau outer v_tau
r_t     = q_t^T S_(t-1)
S_t     = gamma S_(t-1) + k_t outer v_t
```

The current token is excluded because the read uses `S_(t-1)` and the write happens afterward. `gamma` adds exponential retention:

```text
S_(t-1) = sum_(tau<t) gamma^(t-1-tau) k_tau outer v_tau
```

Substitution recovers the weighted history sum. This is why an explicit-history oracle and the fixed-state accumulator can agree even though only one stores past entries separately.

**Code trace:** `CausalLinearAttentionState.read()` then `.write()` in `src/engine/attention.ts`. The unit test compares it with `explicitHistoryRead()`.

**Complexity statement:** for fixed key/value dimensions, state memory does not grow with sequence length. Say “constant in sequence length,” not “free” or “constant in model width.” The state has `keyDimension x valueDimension` entries.

## Derivation 2: mapping into the toy BDH cell

Use these shapes:

```text
u_t, x_t, y_t : R^N
E             : R^(N x D)
Dx, Dy        : R^(D x N)
v*_t          : R^D
sigma_t       : R^(N x D)
```

The implemented cell computes:

```text
v*_t    = u_t E
x_t     = scale [u_t + ReLU(v*_t Dx)]
read_t  = x_t^T sigma_(t-1)
y_t     = ReLU(read_t Dy) elementwise-multiply x_t
sigma_t = gamma sigma_(t-1) + x_t outer v*_t
```

Read before write preserves causality. `sigma` is stored as `N x D`; `sigma Dy` is an `N x N` effective connection matrix of rank at most `D`, which Lab 1 renders.

The trained projection matrices would be slow parameters. Here they are seeded random arrays and remain fixed during a session. `x`, `y`, and `sigma` are fast state. Reset zeroes activity/state but does not regenerate the projections.

**Important precision:** in this implementation `u_t` is positive and sparse, and `x_t`/`y_t` are non-negative. `v*_t` may be signed because `E` is signed; therefore the raw write and effective synapse matrix may be signed. Do not claim every matrix update in this toy is non-negative.

## Derivation 3: what Lab 2 actually compares

For each earlier position, the lab forms one shared raw similarity:

```text
a_tau = u_t dot x_tau
```

The top profile is:

```text
soft_tau = softmax(a_tau / 0.1)
```

The bottom profile is:

```text
syn_tau proportional-to max(0, a_tau gamma^(t-1-tau))
```

Lab 2 does not plot these per-position profiles. It computes the full value output in two ways: a fixed recurrent state and an explicit sum over prior keys and values. Their maximum absolute component error is the live correctness check. A third output applies causal softmax to the same history as a deliberately non-equivalent contrast.

Why retain history in the oracle? It gives a transparent reference implementation, while the fixed aggregate `sigma` does not retain exact token identities or expose a free attention map. `BDHCell.retrievalWeights` is empty by design. What is proven locally is equality between two implementations of the same causal decayed linear-attention operation within floating-point tolerance—not equality to softmax or to every detail of full BDH.

## Derivation 4: why interference appears in Lab 3

The experiment stores `L` associations:

```text
sigma = sum_i k_i outer v_i
```

Cue association `j` using the code's row-vector convention:

```text
vhat_j = k_j^T sigma
       = (k_j dot k_j) v_j + sum_(i != j) (k_j dot k_i) v_i
```

Unit keys make the first coefficient approximately one. The remaining sum is cross-talk. With few near-orthogonal random keys, cross-talk is small. As load grows relative to dimension, cross terms accumulate, cosine to the true value falls, and wrong-value similarity becomes more prominent.

The code calls recall successful when the retrieved vector's nearest stored value is the correct value. The plotted curve averages three deterministic seeds and also reports the strongest incorrect cosine competitor. These choices are disclosed and reproducible.

This derivation supports a qualitative failure mode for fixed associative state. It does not establish the capacity of a trained nonlinear BDH model.

## Control predictions

| Action | Prediction | Reason |
|---|---|---|
| Lab 1: lower `gamma` to `0` | Previous state is removed at every write; only the current outer product survives after a step. | `sigma <- gamma sigma + write`. |
| Lab 1: raise `gamma` to `1` | Writes accumulate without decay; matrix magnitude/cross-talk can build. | Old state is fully retained. |
| Lab 1: raise update scale | `x` becomes larger; the outer-product write generally changes magnitude nonlinearly through feedback. | Scale enters `x`; `x` also affects readout and later `y`. |
| Lab 1: change speed | Visual time changes, mathematical recurrence does not. | Speed only changes timer delay. |
| Lab 2: lower `gamma` | Older associations contribute less to both linear implementations. | Older terms carry higher powers of a number below one. |
| Lab 2: change query position | The causal history and all three output vectors change. | Only positions before `t` contribute. |
| Lab 3: increase `d` at the same load | Random keys tend to be more separable; cross-talk should usually fall. | Higher-dimensional random unit vectors are closer to orthogonal. |
| Lab 3: push load above `d` | Recall should degrade on average. | More cross terms share a fixed matrix. |

## Anticipated questions

### “Is this the official BDH implementation?”

No. It is an independent educational toy inspired by the cited BDH-GPU equations. It has one layer/head, random projections, synthetic embeddings, scalar decay, and no trained checkpoint.

### “Did you run BDH or BDH-CQ?”

No. No live official checkpoint was run. The PS-1 brief explicitly permits published equations, public code, documented evaluations, and clearly labeled precomputed material when checkpoints are unavailable. Paper metrics are author-reported.

### “Where is the model memory?”

Cross-token state inside `BDHCell` is the fixed `sigma` array (`N x D`). The display computes `sigma Dy`. Lab 2's explicit-history oracle separately keeps prior pairs to verify the recurrent result; that history is not part of the fixed-state cell.

### “If Lab 2 stores history, doesn't that invalidate O(1) memory?”

It would invalidate an O(1) claim about the entire visualization. The precise claim is that the **recurrent state** is fixed in sequence length. Lab 2 deliberately pays `O(t)` history for its reference oracle. This is disclosed; an inference implementation would omit that oracle.

### “Is synaptic retrieval equal to softmax attention?”

No. The derivation applies to unnormalized causal linear attention, whose sums can be reordered into a recurrent matrix. Lab 2 verifies that identity directly. Softmax is displayed separately to make the boundary visible: its exponential normalization changes the operation.

### “What does maximum absolute error establish?”

It checks that every output component from the recurrent implementation matches the explicit-history oracle up to floating-point rounding. It does not establish semantic quality, trained-model performance, or equivalence to softmax.

### “Why call the update Hebbian?”

It is an outer product of co-active signals: the current neuron activity and a low-rank message. Each state entry changes using local pre/post-like factors, matching the Hebbian-like form highlighted by the BDH sources. “Hebbian-like” is the careful phrase.

### “Are activations and synapses non-negative?”

`x` and `y` are non-negative in this toy. The low-rank message and effective synapse entries can be signed. The canvas uses green/red precisely because `sigma Dy` may excite or inhibit. Do not collapse these into one claim.

### “Why is the effective matrix `N x N` if stored state is `N x D`?”

The display multiplies `sigma (N x D)` by `Dy (D x N)`. The resulting effective matrix is `N x N` but low rank; it is computed for visualization rather than stored as the recurrence state.

### “Does fixed size mean infinite memory?”

It permits an unbounded number of processing steps without adding one state slot per token. It does not guarantee lossless retention. Decay and interference are the central counterexample shown in Lab 3.

### “What is ground truth in each lab?”

- Lab 1 exposes internal arrays directly; there is no external semantic label.
- Lab 2's reference is an explicit history sum for the same decayed linear-attention equation; maximum absolute error compares its output with the recurrent read. Softmax is a separate contrast.
- Lab 3's known target is the generated `v_i`; nearest-value identity determines recall, while strongest-wrong cosine shows the closest competitor.

### “Why are Lab 3 results reproducible?”

Keys/values use deterministic `mulberry32`; load and seed determine the vectors. Curve points average seeds `42`, `59`, and `76`. The cue button uses seed `1234`.

### “What is actually precomputed?”

Only the static BDH-versus-Transformer comparison table is a prewritten synthesis of cited properties. Labs 1–3 compute in the browser. There is no `public/data` result file and no cached checkpoint output.

### “Is the public demo ready?”

The static build is locally ready, but no public artifact URL or public repository is verified yet. Both are remaining submission deliverables.

### “How did AI contribute?”

OpenAI Codex assisted drafting/editing code, prose, research organization, tests, and docs. The team owns verification and must explain every line and citation. AI output is not used as scientific evidence.

## Live code walk-through route

If asked to trace one token, open these in order:

1. `src/labs/lab1.ts`: `tick()` chooses a token and calls `cell.step(token)`.
2. `src/engine/bdh.ts`: `embedding()` creates/reuses the seeded sparse `u`.
3. `BDHCell.step()`: compute `v*`, then `x`, then read old state, then compute `y`, then write new state.
4. `src/engine/attention.ts`: `read()` and `write()` show the exact matrix loops and causal order.
5. `src/labs/lab1.ts`: `draw()` passes returned arrays to the renderers.
6. `src/viz/neuronCanvas.ts` and `synapseCanvas.ts`: values become pixels; no hidden model/API call exists.

For a proof test, show the fixed-vs-explicit-history test in `tests/engine.test.ts`, then run `npm test`.

## Claims to avoid

- “We trained BDH.”
- “We ran BDH-CQ or reproduced ARC-AGI.”
- “Our toy proves BDH performance or biological plausibility.”
- “The whole artifact uses O(1) memory.” Lab 2 keeps history for explanation.
- “The fixed state retrieves exact old tokens.”
- “Lab 2 is exactly softmax.”
- “Every synaptic value is non-negative.”
- “A public URL/repository exists” until both are published and verified.

## Final pre-defense check

```bash
npm ci
npm test
npm run build
```

Then open the production preview, run the sixty-second claim test, and have one presenter explain the equations while another traces the corresponding loops. If a result is from a paper, say “author-reported” before quoting it.
