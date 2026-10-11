---
license: cc-by-4.0
pretty_name: Prior-shift diagnostic for LLM forgetting
tags:
- forgetting
- calibration
- post-training
- continual-pretraining
- language-models
size_categories:
- n<1K
---

# Prior-shift diagnostic

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23304208.svg)](https://doi.org/10.5281/zenodo.23304208)

When a derived language model (instruction-tuned, preference-tuned, or continued-pretrained) gets
worse on text its base model was trained on, how much of the loss increase can a context-free
correction of its logits, a token bias and/or a temperature, take back?

This deposit holds the diagnostic (`code/prior_shift.py`), per-document results for 14 released
base/derived pairs, and a CPU script (`reproduce.sh`) that re-runs the synthetic tests and
recomputes every point estimate from the per-document sums.

## Result

The scoring text is one million tokens of web text from one DCLM shard of the OLMo-2 pretraining
mix. For the OLMo-2 pairs it is probably training data, since OLMo-2 saw that mix about once.
SmolLM2's model card lists DCLM among its sources, so for it the text comes from the same source
family. For Qwen2.5 it is a proxy for web pretraining data. Whether any particular document was
seen is unknown for every model.

- **The context-free biases fitted here recover little of the loss increase.** The main
  out-of-sample measurement is a bias fitted on the 4,096 most frequent tokens. Its symmetric
  share is at most 0.098, on Qwen2.5-7B-Instruct, and below 0.07 on every other pair. A one-step
  closed-form bias fitted on the scoring text itself gives at most 0.158, on the same pair. That
  in-sample figure is not an upper bound: an optimally fitted bias can recover more than one step
  does. So the result is that these fitted bias corrections stay below 0.20 on all 14 pairs, not that
  no bias could do better. On the headline pair the top-4K fit hit its 20-iteration cap while
  its held-out FIT-slice loss was still falling by about 2·10⁻⁵ nats per step. The measured EVAL
  share is 0.098; what further iterations would do on EVAL was not measured.
- **Adding a temperature recovers more, most of all in the OLMo-2 preference-tuned models.** Most
  derived models are overconfident on this text: the fitted T is up to 1.22, against 1.00–1.01 for
  their bases (for Qwen2.5-Coder-1.5B the difference is within the search resolution). Within
  OLMo-2, the bias-plus-temperature share S_sym^aff rises from SFT to DPO: OLMo-2-1B is at 0.06
  after SFT and 0.16 after DPO, and
  OLMo-2-7B-Instruct reaches 0.26 (0.36 with the top-4K bias), where the top-4K bias alone gives
  0.04. The gap between the two fits is an incremental contrast between fitting procedures, not a
  decomposition: no temperature-only correction is fitted. This is two within-family contrasts
  with one recipe. Other instruction-tuned models are far lower (SmolLM2-1.7B-Instruct 0.021,
  Qwen2.5-1.5B-Instruct 0.018), so it is not a general property of preference tuning.
- **Continued pretraining on math forgets a lot and recovers almost nothing.** Qwen2.5-Math loses
  0.49 (7B) and 0.80 (1.5B) nats/token. Every out-of-sample correction recovers ≤ 0.031 of it, and
  the in-sample one-step closed form ≤ 0.055. The Coder models lose less (0.13), with similarly
  near-zero shares.

The full table is `data/summary.md`, with 95% document-bootstrap intervals in `data/summary.json`.
These intervals resample EVAL documents with the fitted corrections held fixed. They are much
narrower than the spread from fitting choices: for example, the closed-form share of
Qwen2.5-7B-Instruct ranges from 0.077 to 0.120 over the three smoothing strengths in `result.json`,
and the top-4K bias interval for that pair reaches 0.104.

## Why the closed-form bias is not the headline

The smoothed closed-form correction (`cf`, METHODS.md) gives every token unseen in FIT a fixed
negative bias. At the reported smoothing strength this makes every base model worse on EVAL
(R_cf_old from −0.022 to −0.030 nats/token), and 12 of 14 derived models too. Its symmetric share
therefore partly measures which model the smoothing hurts more. The same tail penalty is a likely
reason the full-vocabulary fitted bias early-stopped at b = 0 on every model. The top-4K bias moves
only frequent tokens. Its fit stays at b = 0 for all six bases, so for it S_sym equals S_card. The
in-sample one-step figure comes from `jensen_in_sample_E` in each `result.json`.

## A caveat for any "a simple correction explains X%" claim

How much an affine correction recovers does not identify what caused the loss. In one synthetic
known-answer case, the derived model's output weights get random, context-dependent damage, and
the construction adds no global bias and no temperature. The fitted bias-plus-temperature
correction still recovers 0.29 of the loss increase, and the bias alone 0.08
(`code/test_prior_shift.py`, case `ctx`; `reproduce.sh` prints both). The damage changes the
model's calibration as a side effect, and a temperature partly undoes that.

This is one construction at one damage scale, not a calibrated threshold. It is a counterexample to
reading a combined share of that size as evidence for a global prior or temperature change. It
does not imply that a genuine temperature change must show up in the bias share: the `temp` case
is a pure temperature change, which the affine correction recovers almost entirely and the bias
alone does not.

## Quantities

For a base model (old) and a derived model (new) sharing a tokenizer, scored on the same EVAL
documents, Δ = CE_new − CE_old (nats/token). For a correction x fitted separately to each model on
FIT documents, with R^x the cross-entropy it removes:

    S_card^x = R_new^x / Δ                  share recovered on the derived model
    S_sym^x  = (R_new^x − R_old^x) / Δ      net change in what the correction recovers

S_sym is reported because S_card overstates the share when the base itself gains from the
correction on this text: the synthetic case `proxy` shows S_card 0.47 against S_sym 0.30. S_sym is
a difference of two gains. It can cancel when the two models gain for different reasons, e.g. a
bias the derived model removed and a new one it introduced. A small S_sym is therefore not evidence
that no new prior error exists. For the top-4K bias, R_old = 0 on every base, so no cancellation
occurs there. Corrections, fitting and statistics are in `METHODS.md`.

## Pairs

| Group | Base | Derived | Regime |
|---|---|---|---|
| g1 | OLMo-2-0425-1B | -SFT, -DPO, -Instruct | post-training |
| g2 | OLMo-2-1124-7B | -SFT, -DPO, -Instruct | post-training |
| g3 | Qwen2.5-7B | -Instruct, Qwen2.5-Math-7B, Qwen2.5-Coder-7B | post-training; continued pretraining |
| g4 | Qwen2.5-1.5B | -Instruct, Qwen2.5-Math-1.5B, Qwen2.5-Coder-1.5B | post-training; continued pretraining |
| g5 | SmolLM2-1.7B | -Instruct | post-training |
| g6 | OLMo-2-0425-1B at branch `stage1-step1907359-tokens4001B` (commit 9d3e43659f00, as listed on 2026-10-09) | OLMo-2-0425-1B (final) | mid-training anneal |

Every model is pinned to a Hugging Face commit; the full SHAs are in each `result.json`
(`revisions`) and `inputs.json` (`args`).

## Layout

    code/prior_shift.py        the diagnostic (GPU; one base, one or more derived models)
    code/summarize.py          table and descriptive flags from a results directory
    code/test_prior_shift.py   synthetic known-answer tests (CPU, about a minute on an idle machine)
    data/<group>/inputs.json   document indices, SHA-256 of each document's text, token counts
    data/<group>/<model>/result.json   estimates, intervals, fitted temperatures, bias-fit traces
    data/<group>/<model>/docsums.npz   per-document loss sums under every correction, both models
    data/<group>.log           run log
    data/summary.{json,md}     the full table of results
    verify_data.py             recomputes the point estimates from docsums.npz
    test_verify_data.py        negative controls: verify_data.py must reject corrupted inputs
    reproduce.sh               runs all checks on a CPU

The data contain no corpus text: documents are identified by index and SHA-256 within one public
shard (`inputs.json`, field `data`). The fitted bias vectors are not saved.

## Checking this deposit

    ./reproduce.sh     # Python 3 with numpy and torch; a few minutes on a CPU

It does four things: verifies the checksums; runs the synthetic tests and prints the values cited
above; recomputes every point estimate in each `result.json` from the per-document sums, checking
that all 14 pairs are present and that the sums are finite and consistent, and shows that the
verifier rejects corrupted inputs; and regenerates the
summary table. It exits non-zero on any failure.

It does not recompute the confidence intervals, the fitted temperatures, the in-sample figures or
the fit traces; those need the models. Nor does it tie `data/` to the code, which needs the GPU
re-run below. `code/` is the code that produced `data/`; only docstrings, comments and the test's
import name differ.

## Re-running the models

On a GPU (all pairs ran on one A100-80GB; Colab's resource monitor, read by hand and not
recorded, showed ≈ 21 GB GPU and ≈ 26 GB host memory on the 7B pairs):

    python code/prior_shift.py --base Qwen/Qwen2.5-7B --base_rev d149729398750b98c0af14eb82c78cfe92750796 \
        --new Qwen/Qwen2.5-7B-Instruct --new_rev a09a35458c702b33eeacc393d103063234e8bc28 --out out/g3

The run used Python 3.13 and torch 2.11 (recorded in each `result.json`) and transformers 5.18 (not
recorded), with numpy, zstandard and requests. The sweep took about 13.5 A100-hours, mostly the
temperature search. Model revisions are pinned. The data shard is read from the dataset's `main`
branch, not a pinned commit. Every document's SHA-256 is in `inputs.json`, but the code does not
compare a re-run against it, so drift has to be checked by hand. Results are deterministic up to
GPU floating-point nondeterminism.

## What is not claimed

- **Scope.** One web shard, one million EVAL tokens, 14 pairs from three model families, all at or
  below 7B. Nothing here speaks to larger models, to other old-distribution text, or to benchmark
  forgetting.
- **No bound on bias recoverability.** The shares are those of the fitted procedures (smoothing,
  early stopping, top-4,096 support). The in-sample one-step figure is not a ceiling. A
  differently fitted bias might recover more. K = 4,096, the IPF step (undamped, no line search)
  and the early-stopping split were not varied. That every base fit stays at b = 0 is a property of
  this procedure, not of the models. The symmetric in-sample share (0.158 at most) differs from the
  per-model in-sample share S_card, which is large (e.g. 0.78 on Qwen2.5-7B-Instruct), because
  the base gains in sample too.
- **Excluded tokens are not measured.** Special and added tokens (EOS, chat-template tokens) are
  removed from the softmax, so all losses are conditional on the regular vocabulary. A prior shift
  onto excluded tokens, e.g. more EOS mass, would raise a model's full-vocabulary loss and be
  invisible here.
- **A small S_sym does not rule out an introduced prior error** (see Quantities). Format- and
  position-specific shifts fall on the context-dependent side of this split by definition.
- **The flags in `code/summarize.py` are descriptive thresholds.** The 0.20 bias bar and the 0.39
  combined-share bar (0.29 from the synthetic `ctx` case plus 0.10) are fixed numbers, not
  calibrated tests. No flag firing is not evidence against a mechanism.
- **The affine fit is not a certified optimum.** The temperature is chosen by golden-section search
  over early-stopped inner fits; the profile is not shown to be unimodal, and T = 1 is not
  explicitly among the candidates.
- **Same tokenizer, same targets.** Base and derived models are scored on the base tokenizer's ids.
  Tokenizer agreement is checked on one probe sentence, and per-document target counts are taken
  from the base model, not cross-checked.
- **The scoring text is not held out for every model.** For OLMo-2 it is likely training data.
- **The temperature finding is not new in kind.** Post-training is known to hurt calibration on
  benchmarks; this measures the same effect as a share of web-text loss.

## Citation and licence

Archived on Zenodo: [10.5281/zenodo.23304208](https://doi.org/10.5281/zenodo.23304208) (this
version; all versions: 10.5281/zenodo.23304207). See `CITATION.cff`. **Everything, data included, is CC BY 4.0** (`LICENSE-DOCS`). The code
(`code/`, `reproduce.sh`, `verify_data.py`, `test_verify_data.py`) is additionally offered under MIT
(`LICENSE`); GitHub's sidebar shows only the latter.

Source data: documents from one DCLM shard of `allenai/olmo-mix-1124` (ODC-BY; its DCLM-Baseline
part is CC BY 4.0). No corpus text is redistributed here, only document indices, SHA-256 hashes,
token counts and per-document loss sums.
