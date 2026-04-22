# Persona Vectors — Plain-language summary

This is a short reading companion for Anthropic's paper
[Persona Vectors: Monitoring and Controlling Character Traits in Language Models](https://arxiv.org/abs/2507.21509)
(Chen, Arditi, Sleight, Evans, Lindsey, 2025). It covers what you need before
touching notebook 01. Read the paper's Sections 1–3 alongside this.

## The claim in one sentence

Character traits of a chat model's Assistant persona — like "evil", "sycophancy",
"hallucination" — correspond to **linear directions in residual-stream
activation space**, and those directions can be used to *monitor, steer, predict,
and filter* personality shifts.

## The pipeline (Figure 2)

Given a trait name + natural-language description, the authors run an automated
four-step pipeline:

1. **Generate artifacts with a frontier LLM:** 5 pairs of contrastive system
   prompts (one that elicits the trait, one that suppresses it), 40 evaluation
   questions split into extraction + held-out sets, and an LLM-judge rubric that
   scores a response 0–100 for how strongly it exhibits the trait.

2. **Generate rollouts:** under each (positive/negative) × (system prompt) ×
   (question) condition, sample 10 responses from the target model.

3. **Filter by judge score:** keep positive rollouts scoring > 50 (definitely
   trait-exhibiting) and negative rollouts scoring < 50 (definitely not).

4. **Extract persona vectors:** for each layer, average residual-stream
   activations over the response tokens for each group; the persona vector is
   `mean(positive) − mean(negative)`. One candidate vector per layer; pick the
   most informative layer by a small steering sweep on held-out questions.

## The four applications (Figure 1)

Once you have `v_ℓ` for trait `t` at layer `ℓ`:

### Monitoring (Section 3.3, Figure 4)

Project the activation at the **last prompt token** onto `v`. Paper reports
Pearson `r = 0.75–0.83` between that projection and the post-hoc trait-expression
score of the generated response. You can tell *before the model speaks* whether
it's about to exhibit the trait.

### Inference-time steering (Section 3.2, Figure 3)

At every decoding step apply `h_ℓ ← h_ℓ + α · v_ℓ`. Positive α induces the
trait; negative α suppresses it. Typical α range 0.5–2.5. Too-large α degrades
general capability (MMLU drops).

### Preventive steering during finetuning (Section 5.2, Figure 7B)

During SFT, keep `h_ℓ ← h_ℓ + α · v_ℓ` active at every training forward pass.
Intuition: the model no longer has to shift its activations along `v` to fit the
data (you've already shifted them for it), so gradient pressure is "spent"
elsewhere. Post-training trait expression stays low *and* MMLU is preserved
better than with post-hoc inference-time suppression.

### Dataset screening (Section 6)

For each training sample, project its response activation onto `v`. High
projection → sample is likely to reinforce the trait if trained on. Catches
problematic samples that LLM-based data filters miss.

## Models studied

Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct. Three primary traits (evil,
sycophancy, hallucination) with extensions to optimism, humor, and others in
the appendix.

## Useful intuitions

- **Why a linear direction works at all:** similar to the "linear representation
  hypothesis" in interp — high-level concepts in LLMs often collapse to linear
  subspaces. This is why difference-of-means (a very crude probe) gives a usable
  vector. Persona traits are another instance of that structure.

- **Why mid-layers usually win:** early layers handle syntactic/token features;
  final layers are committed to specific tokens. Trait-level semantics live in
  the middle. For 7B models this is around layer 15–20.

- **Why preventive steering works:** finetuning-induced persona shifts
  correlate extremely well (r up to 0.97) with activation shifts along `v`. If
  you pre-move the activations along `v`, fitting the data no longer requires
  shifting them further — the model's weights end up closer to the base.

## What's *not* in the paper (yet)

- How to handle traits that are only partially linear, or that require
  compositional directions.
- How robust vectors are across finetunes, or across model scales.
- Rigorous study of cross-trait interference (though they note some traits
  correlate along the same axis).

These are all candidates for your capstone.
