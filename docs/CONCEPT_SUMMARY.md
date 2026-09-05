# Synapse Memory Lab: concept summary

## The central idea

Causal **linear** attention can be evaluated in two mathematically equivalent ways. An explicit-history implementation stores every past key–value pair and, before processing token `t`, computes

\[
y_t=\sum_{\tau<t}\gamma^{t-1-\tau}(q_t^\top k_\tau)v_\tau.
\]

A recurrent implementation instead maintains the fixed-shape matrix

\[
S_t=\gamma S_{t-1}+k_t v_t^\top,\qquad y_t=q_t^\top S_{t-1}.
\]

Expanding the recurrence gives the history sum exactly (up to floating-point effects). The read-before-write order makes it strictly causal: the current token cannot retrieve itself. With key width `d_k` and value width `d_v`, explicit history grows as `O(t(d_k+d_v))`, whereas the `d_k × d_v` state stays `O(d_kd_v)` in sequence length.

This identity concerns linear/kernel attention. It is **not** an exact reformulation of ordinary softmax attention. Softmax normalizes exponentiated pairwise scores across the available positions; a linear-attention state uses a chosen feature map or dot-product kernel, and may require a separate recurrent normalizer. Similar retrieval patterns do not establish softmax equivalence [3].

## Three views of the same computation

| View | Stored during inference | Read | Main benefit | Main cost |
|---|---|---|---|---|
| Explicit linear-attention history | All key–value pairs | Weighted sum over positions | Inspectable per-token contributions | Memory and serial decoding work grow with context |
| Recurrent linear attention | Matrix `S_t` | `q_t^T S_{t-1}` | Fixed shape; constant memory in sequence length | Past writes are superposed; identity of individual tokens is not retained |
| BDH/BDH-GPU interpretation | Neuron activity plus evolving synaptic state | Activity-dependent state readout | Gives the matrix update a local, Hebbian/synaptic interpretation | Claims beyond the equations depend on a young, author-led evidence base |

The outer product `k_t v_t^T` writes an association: coordinates active together change their shared state. This resembles a fast-weight or Hebbian associative memory. Decay or gating can reduce stale information, but attenuates older contributions. Without enough separation between keys, multiple writes overlap and interfere. Fixed shape therefore removes sequence-length memory growth; it does not create unlimited, lossless memory.

## Where BDH and BDH-CQ fit

The 2025 **Dragon Hatchling (BDH)** paper develops a graph-based model and a tensor-friendly BDH-GPU formulation. Its recurrent matrix state is updated with an outer-product-like term, then read using current positive neuron activity. The authors interpret this evolving state as synaptic plasticity and report sparse positive activations, concept-sensitive synapses, and GPT-2-like scaling comparisons [1]. The lab isolates this memory mechanism in a small educational cell; it is not a trained reproduction of the full architecture.

The 2026 **BDH-CQ** paper extends the family toward in-context learning and recurrent latent reasoning: demonstrations update contextual memory, and the model iterates in latent space before producing an answer [2]. Its ARC-AGI-1 accuracy and cost figures are useful early evidence, but are author-reported results from a new preprint. Public descriptions justify the high-level role above; they do not justify inventing proprietary implementation details.

## Evidence and maturity

- **Established algebra:** explicit-history and recurrent-state evaluation are equivalent for the specified causal linear kernel. This follows directly by expanding the recurrence and is independently testable.
- **Established research lineage:** fast weights and recent linear-attention systems use matrix-valued recurrent states and outer-product updates. Gated Linear Attention explicitly presents this view and also notes that linear attention can underperform standard attention [3].
- **Author-reported evidence:** BDH’s biological interpretation, sparsity, interpretability, scaling results, and BDH-CQ’s benchmark/cost results come from their respective authors’ preprints and repositories.
- **This lab’s evidence:** deterministic unit tests can verify causality, shape invariance, and numerical agreement with an explicit-history oracle. Toy recall experiments illustrate attenuation and interference, but do not measure the capacity of trained BDH.
- **Maturity assessment:** linear-attention recurrence is well established; modern gated/fast-weight variants are active research [3,4]; BDH is an early 2025 preprint with public code; BDH-CQ is a very recent 2026 preprint [1,2]. Both are promising research prototypes, not yet a broadly replicated production standard.

## Strengths and limitations

The fixed-state formulation enables bounded inference memory, streaming operation, local updates, and a state that can be inspected as associations rather than a list of tokens. Its limitations are equally structural: collisions, recency bias under decay, finite precision, order-sensitive updates, and weaker direct attribution to particular past positions. Quality depends on learned representations, normalization, gating/decay, state dimension, and training—not on the recurrence alone.

## Recent primary sources

1. Kosowski et al., **“The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain”** (2025), [arXiv:2509.26507](https://arxiv.org/abs/2509.26507).
2. Engdahl et al., **“BDH-CQ: In-Context Learning with Recurrent Latent Reasoning”** (2026), [arXiv:2608.09888](https://arxiv.org/abs/2608.09888).
3. Yang et al., **“Gated Linear Attention Transformers with Hardware-Efficient Training”** (2023; revised 2024), [arXiv:2312.06635](https://arxiv.org/abs/2312.06635).
4. Sun et al., **“Learning to (Learn at Test Time): RNNs with Expressive Hidden States”** (2024), [arXiv:2407.04620](https://arxiv.org/abs/2407.04620).

