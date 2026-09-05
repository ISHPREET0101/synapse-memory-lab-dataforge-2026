# Four-minute final presentation

## 0:00–0:25 — Hook

“Transformers usually remember a session by keeping one key and value per token. Synapse Memory Lab asks a narrower, testable question: when attention is linear, can that growing history be reorganized into a fixed-shape associative state? Yes—and the price is visible interference.”

State the evidence boundary immediately: this is an independent, deterministic educational toy grounded in the cited BDH equations, not a trained or official Pathway checkpoint.

## 0:25–1:20 — Lab 1: make memory visible

Point to the 96×12 recurrent state and single-step twice.

- Set `gamma = 1`: predict that earlier writes remain and new outer products accumulate.
- Reset, set `gamma = 0`, single-step twice: predict that each new write replaces the retained state.
- Point out that the square 96×96 heatmap is derived as `rho Dy`; it is not the stored recurrent allocation.

Say: “The current activity reads the old state before writing. That ordering is the causal contract.”

## 1:20–2:15 — Lab 2: prove the algebra

Move query position and decay. Point to green and blue bars and read the displayed maximum absolute error.

“Green uses only the fixed 12×8 matrix. Blue explicitly sums every earlier key/value pair. They implement the same decayed linear-attention equation, so they overlap within floating-point noise. Orange is softmax and is deliberately exempt: normalization and exponentiation make it a different operation.”

Do not say “BDH equals softmax.”

## 2:15–3:05 — Lab 3: break the abstraction

Select `d = 64`; compare load 20 with load 160.

“These are random key/value associations in a separate generic memory. Correct means the intended value is the nearest stored value by cosine similarity. As more writes share one matrix, the strongest wrong match becomes a real competitor. This demonstrates a failure mode; it is not a calibrated BDH capacity curve.”

## 3:05–3:40 — Map to BDH

Use the BDH module to identify:

1. current same-token activity `x`;
2. low-rank message `v*`;
3. compressed recurrent state `rho`;
4. read-before-write causality; and
5. frozen slow projections versus evolving fast state.

Label the BDH and BDH-CQ metrics as author-reported before quoting any number.

## 3:40–4:00 — Close

“The result is not ‘infinite memory’ and not ‘softmax without a cache.’ It is a precise engineering trade: fixed sequence-length state through algebraic reordering, with inspectable decay and interference. The site lets a learner predict, test, and falsify that claim in under a minute.”

If time remains, open `tests/engine.test.ts` and show the read-before-write, fixed-byte-size, whole-state decay, and recurrent-versus-history equality tests.
