# Methods

Everything below is what `code/prior_shift.py` does; function names are given so each step can be
checked against the code.

## Data and splits

Documents are read in file order from one shard of the DCLM web portion of the OLMo-2 pretraining
mix (`inputs.json`, field `data`). Each document is tokenised without special tokens, truncated
to 2,047 tokens, prefixed with the tokenizer's BOS token (EOS if it has none), and skipped if
shorter than 16 tokens. Whole documents go to FIT until it holds 10⁶ target tokens, then to EVAL
until it holds 10⁶ (`build_splits`). Splits depend only on the tokenizer, so every pair in a group (one base, several
derived models) is scored on identical documents, and groups that share a tokenizer share splits.

The fitted corrections (bias, aff and their top-4K variants) are fitted on the first 90% of FIT
documents and early-stopped on the last 10%. The closed form uses all FIT positions.

## Logits

Final hidden states h (after the final norm) are cached in bf16 for every position, and logits are
recomputed as z = h Wᵀ with fp32 accumulation and TF32 matmuls (`cache_hidden`, `Logits`). Because
h and W are bf16-valued, TF32 loses nothing on the inputs; the accumulated dot products are
ordinary finite-precision fp32 results. They need not equal a model's own bf16 logits bit for bit. Logits are restricted to the *regular* vocabulary: ids below
`len(tokenizer)` that are not special or added tokens in any tokenizer of the group
(`regular_vocab_mask`). Other ids get −∞, and targets outside the regular vocabulary are dropped. Every loss is
therefore conditional on the regular vocabulary: probability mass moved onto excluded tokens is not
measured.

Both models are fed the base tokenizer's ids. Tokenizer agreement is checked on one probe
sentence only. Per-document target counts are taken from the base model's pass. `inputs.json`
records `len(ids)` per document, which includes the prefix token. The scored target count is one
less, minus any targets outside the regular vocabulary.

## Corrections

Each is fitted separately for the base and the derived model, on FIT only. p̄ is the model's mean
predictive distribution over FIT positions, c the FIT target counts, and N their total.

- **id**: no correction.
- **cf** (closed form): b = log q_s − log p̄ with q_s = (c + β p̄)/(N + β), β ∈ {10⁴, 10⁵, 10⁶};
  β = 10⁵ is reported (`closed_form_bias`). It has the form of the first step of the bias fit
  below (Kneser et al. 1997, first-iteration approximation). It is not literally that step: it uses
  all FIT documents and a fixed β, whereas the fit uses the training subset and β′. It has the form of a prior-ratio correction (cf. Saerens et al. 2002), but
  its denominator is the model's own mean prediction on FIT text rather than a training prior.
  Every token unseen in FIT gets the fixed bias log(β/(N + β)), e.g. −2.4 at β = 10⁵ with N ≈ 10⁶.
  This tail penalty makes the correction hurt base models on EVAL (README.md).
- **bias**: iterative proportional fitting, b ← b + log t − log p̄_b, where p̄_b is the mean
  predictive distribution under bias b and t = (c + β′ p̄)/(N + β′) with β′ = 0.1 N (`smoothed_target`,
  `fit_bias_ipf`). Each target activates one bias coordinate, so this is generalised iterative
  scaling with unit step. Its fixed point matches the model's mean prediction to t (minimum
  discrimination information adaptation, Kneser et al. 1997). The iterate with the lowest held-out
  FIT cross-entropy is kept: at most 20 iterations, stopping after 3 without improvement, and b = 0
  if no step helps. Some top-4K fits reached the 20-iteration cap while still improving, by less
  than 5·10⁻⁵ nats per step, so they are truncated rather than early-stopped. The fitted bias
  vectors are not saved.
- **aff**: temperature plus bias, softmax(z/T + b). log T is found by golden-section search on
  [log 0.25, log 4] with 14 evaluations; at each T the bias is refitted as above, and the T with the
  lowest held-out cross-entropy is kept (`fit_affine`). The search is not a certified optimum: the
  profile is not shown to be unimodal, and T = 1 is not explicitly among the candidates. This has the functional form of
  bias-corrected temperature scaling (Alexandari et al. 2020), which fits it by maximum likelihood
  on held-out labels; here the bias is fitted toward a smoothed target with early stopping.
- **bias_top4k, aff_top4k**: as bias and aff, but only the 4,096 most frequent target tokens of the
  FIT training documents may move; all other coordinates stay 0. Restricting the support is a
  second form of regularisation. On every model the full-vocabulary bias early-stopped at b = 0:
  one step raised held-out loss, plausibly because of the tail penalty above together with
  document-level unigram variation.

## Statistics

For each correction x, the EVAL cross-entropy is summed per document for both models
(`score`). Each model is scored under its own fitted correction, and `stats_from_docsums` forms Δ, S_card^x and S_sym^x (see `README.md`) as ratios of
token-weighted means. The 95% intervals are percentiles over 2,000 resamples of EVAL documents with
replacement, with both models resampled together. Point estimates use all documents. The
fitted corrections are held fixed, so the intervals do not include fitting variability.

`result.json` also records per-position-bucket Δ (positions 1, 2–8, 9–64, 65–512, ≥ 513), the
IPF traces of the bias-only fits, the temperature grid of the affine fits (not their inner IPF
traces), and a correctness check on EVAL: the in-sample
closed-form gain must be at least KL(q_E ‖ p̄_E) (Jensen; `jensen_in_sample_E`). It held for every
model. The check is a lower bound on one model's one-step in-sample gain. It is not an upper
bound on what any bias could recover. Kneser et al. distinguish the first-iteration approximation
from the converged iterative fit, and the converged in-sample fit was not computed.

## Validity and flags (`code/summarize.py`)

A pair is valid if Δ > 0, the lower end of Δ's interval is at least 0.005 nats/token, no estimate
is NaN, and the Jensen check holds for both models. Two descriptive flags are computed. They are fixed thresholds, not calibrated tests:
S_sym^cf ≥ 0.20 or S_sym^bias_top4k ≥ 0.20 (a token-prior component worth pursuing), and
S_sym^aff_top4k ≥ 0.39 (0.10 above the 0.29 measured in the synthetic `ctx` case). No pair raised
either.

## Synthetic known-answer tests (`code/test_prior_shift.py`)

Data are sampled from a known model with V = 400, d = 48, 600 documents of 64 tokens. The derived
model differs from the base by:

| Case | Construction | Required |
|---|---|---|
| bias | a global logit bias | fitted bias and affine recover > 90%; closed form > 60%; Jensen check holds |
| temp | logits doubled (T = 0.5) | affine recovers > 90% with fitted T ≈ 2; bias alone clearly less |
| ctx | random context-dependent damage to W (scale 0.25, against 0.5 for W) | bias shares < 0.1; affine share in (0.15, 0.45), measured 0.29 |
| proxy | scoring text drawn with a prior shift the base lacks | S_card exceeds S_sym by > 0.1; S_sym bias < 0.15 |
| identity | derived = base | Δ = 0 exactly |

The `ctx` case is the source of the 0.29 figure: one damage scale, V = 400, full-vocabulary
affine correction. The tests have equal-length documents, no document-level unigram variation, and an
all-true vocabulary mask, so they do not exercise the masking path or the regime in which the real
full-vocabulary bias fits collapsed to 0. A few vocabulary items do go unseen in FIT. The top-K
corrections (K = 128 in the tests) are computed but have no known-answer assertion; only the
full-vocabulary estimators are tested for recovery.

## References

- Kneser, Peters & Klakow (1997), Language model adaptation using dynamic marginals, Eurospeech.
- Saerens, Latinne & Decaestecker (2002), Adjusting the outputs of a classifier to new a priori
  probabilities: a simple procedure, Neural Computation 14(1).
- Alexandari, Kundaje & Shrikumar (2020), Maximum likelihood with bias-corrected calibration is
  hard-to-beat at label shift adaptation, ICML; arXiv:1901.06852.
