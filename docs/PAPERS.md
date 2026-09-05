# Papers and claim-to-source map

This bibliography uses the recent primary sources already cited by the artifact and the DataForge PS-1 brief. A citation here records what a source is used for; it does not imply that Synapse Memory Lab reproduced the source's experiments.

## Core BDH sources

### [P1] The Dragon Hatchling (BDH-GPU)

A. Kosowski, P. Uznański, J. Chorowski, Z. Stamirowska, M. Bartoszkiewicz, et al. “The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain.” arXiv:2509.26507, 2025.

- Paper: <https://arxiv.org/abs/2509.26507>
- Reference repository: <https://github.com/pathwaycom/bdh>
- Used for: BDH-GPU neuron/synapse equations; Hebbian-like outer-product state updates; fixed recurrent synaptic state; ReLU/non-negative activity; reported sparsity and connectivity observations; reported model-scaling comparisons.
- Artifact boundary: `src/engine/bdh.ts` is a scaled-down, random, untrained, independent implementation with disclosed deviations. The paper's training and evaluation results are not reproduced.

### [P2] BDH-CQ

Pathway. “BDH-CQ: In-Context Learning with Recurrent Latent Reasoning.” arXiv:2608.09888, 2026.

- Technical report: <https://arxiv.org/html/2608.09888v1>
- Used for: the connection between contextual state, demonstration accumulation, in-context adaptation without inference-time parameter updates, recurrent latent reasoning, and the author-reported ARC-AGI-1/cost result displayed in the page.
- Artifact boundary: the project does not implement BDH-CQ, run its checkpoint, reproduce ARC-AGI, or independently verify its cost figures.

### [P3] Pathway derivation chapter

Pathway. “From attention to synapses: deriving BDH.” BDH explainer series, chapter 2.

- Chapter: <https://pathway.com/research/bdh-explainer/bdh-architecture-derivation>
- Used for: the educational derivation from kernel/linear attention to an incrementally updated outer-product state, including the `k = q = x` interpretation.
- Status: primary project-authored technical explanation, but not a peer-reviewed paper. It supplements rather than replaces [P1].

## Recent adjacent primary papers (2022–2026)

### [P4] Mamba

A. Gu and T. Dao. “Mamba: Linear-Time Sequence Modeling with Selective State Spaces.” arXiv:2312.00752, 2023.

- Paper: <https://arxiv.org/abs/2312.00752>
- Used for: a prominent fixed-state sequence-model comparison point.
- Guardrail: the DataForge brief explicitly says not to classify BDH-GPU as an SSM in the Mamba sense. Mamba is context, not an identity claim.

### [P5] Gated Linear Attention

S. Yang, B. Wang, Y. Shen, R. Panda, and Y. Zhang. “Gated Linear Attention Transformers with Hardware-Efficient Training.” arXiv:2312.06635, 2023.

- Paper: <https://arxiv.org/abs/2312.06635>
- Used for: the matrix-state/RNN view of linear attention and gated retention, adjacent to the “state plus outer-product write” mechanism.
- Guardrail: sharing an algebraic pattern does not make GLA biologically interpreted BDH.

### [P6] Test-Time Training layers

Y. Sun, X. Li, K. Dalal, J. Xu, A. Vikram, G. Zhang, Y. Dubois, X. Lu, T. Al-Shedivat, W. Xiong, et al. “Learning to (Learn at Test Time): RNNs with Expressive Hidden States.” arXiv:2407.04620, 2024.

- Paper: <https://arxiv.org/abs/2407.04620>
- Used for: a different answer to the pressure for expressive fixed-size hidden state and test-time adaptation.
- Guardrail: this project does not claim that BDH's Hebbian state update is the TTT optimization rule.

### [P7] Titans

A. Behrouz, P. Zharkov, and A. Mirhoseini. “Titans: Learning to Memorize at Test Time.” arXiv:2501.00663, 2025.

- Paper: <https://arxiv.org/abs/2501.00663>
- Used for: recent test-time memory/plasticity context, particularly memory updates driven by surprise.
- Guardrail: Titans is adjacent work, not evidence for this toy's numerical behavior.

## Claim-to-source ledger

| Claim shown or taught | Source | What this project verifies |
|---|---|---|
| BDH reformulates attention as evolving synaptic memory with Hebbian-like writes. | [P1], [P3] | Implements and visualizes a simplified outer-product state update; does not verify the full trained model. |
| The recurrent state can stay fixed in shape as sequence length grows. | [P1], [P3]; adjacent algebra in [P5] | Unit tests check that the toy's `N x D` state allocation is unchanged after 500 steps. |
| Reads can be expressed as a query against an accumulated outer-product state. | [P3], [P5] | `CausalLinearAttentionState` is tested against an explicit-history sum. |
| BDH has reported sparse non-negative neural activity and heavy-tailed connectivity. | [P1] | Not reproduced; only described as author-reported. |
| BDH-CQ uses contextual/recurrent state for demonstration-based adaptation and latent reasoning. | [P2] | Not implemented; summarized with a source label. |
| BDH-CQ's displayed ARC/cost figures. | [P2] | Not reproduced; author-reported only. |
| Fixed-shape memory may suffer interference. | Mechanistic experiment in this repository; broader context [P5]–[P7] | Directly measured for the synthetic linear associative-memory implementation in Lab 3, not calibrated to BDH. |
| BDH is not simply Mamba-style SSM. | DataForge PS-1 brief; [P1], [P4] for the distinct formalisms | Maintains the distinction in documentation; no comparative benchmark is run. |

## Citation discipline

- “Live” means recomputed by the browser's TypeScript implementation.
- “Synthetic” means generated from fixed token IDs or deterministic pseudorandom vectors, not sampled from a research dataset.
- “Precomputed” on the page refers to the static comparison synthesis, not a cached official checkpoint run.
- “Author-reported” means a number or observation comes from the cited authors and was not independently replicated.
- The absence of a live BDH checkpoint is intentional and permitted by the PS-1 brief; it must remain visible in submission materials.
