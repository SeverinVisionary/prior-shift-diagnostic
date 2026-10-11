#!/usr/bin/env python3
"""Prior-shift diagnostic: how much of a model's loss increase on old-distribution text is
recovered by a context-free correction of its logits?

For a base model and one or more derived models sharing its tokenizer:
  1. Read web documents (one DCLM shard of the OLMo-2 pretraining mix) in file order and
     split them by document into FIT (F) and EVAL (E) by a fixed token budget.
  2. For each model, cache the final hidden states h (bf16) on F and E, and the
     unembedding W. Logits are z = h @ W^T with fp32 accumulation (TF32 loses nothing on
     the bf16-valued inputs).
  3. Fit per model, on F only:
       cf         b = log q_s - log pbar (closed form, q_s Dirichlet-smoothed toward pbar)
       bias       b by iterative proportional fitting toward a smoothed target,
                  early-stopped on the last 10% of F documents
       aff        temperature T by golden-section search on log T over T in [0.25, 4], with the
                  IPF bias refitted at each T
       bias_top4k, aff_top4k   as bias and aff, but only the 4,096 most frequent F tokens
                  may move
  4. Score E for both models under each correction and report
       Delta  = CE_new - CE_old
       S_card = (CE_new - CE_new^x) / Delta              (share recovered on the new model)
       S_sym  = 1 - (CE_new^x - CE_old^x) / Delta       (base given the same correction)
     with document-bootstrap 95% intervals, and the in-sample Jensen bound
     R_cf(E) >= KL(q_E || pbar_E) on E as a correctness check.

Logits are restricted to the "regular" vocabulary: ids below len(tokenizer) that are not
special or added tokens in any of the group's tokenizers. Targets outside it are dropped.

The statistics (`stats_*`, `fit_*`, `analyse`) are pure torch and are exercised on synthetic
known-answer cases by test_prior_shift.py.

usage: python prior_shift.py --base BASE [--base_rev SHA] --new NEW [NEW ...]
                             [--new_rev SHA [SHA ...]] --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import platform
import sys
import time

import numpy as np
import torch

DCLM_SHARD = ("data/dclm/raw/hero-run-fasttext_for_HF/filtered/OH_eli5_vs_rw_v2_bigram_200k_train/"
              "fasttext_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train/processed_data/"
              "global-shard_01_of_10/local-shard_0_of_10/shard_00000000_processed.jsonl.zstd")
DCLM_REPO = "allenai/olmo-mix-1124"
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:8.1f}s] {msg}", file=sys.stderr, flush=True)
POS_BUCKETS = [(1, 1), (2, 8), (9, 64), (65, 512), (513, 10**9)]


# ----------------------------------------------------------------------------- data

def iter_dclm_docs(path_or_url: str):
    """Yield (index, text) from a .jsonl.zstd file, streaming."""
    import zstandard
    if path_or_url.startswith("http"):
        import requests
        raw = requests.get(path_or_url, stream=True, timeout=120).raw
    else:
        raw = open(path_or_url, "rb")
    reader = zstandard.ZstdDecompressor().stream_reader(raw)
    for i, line in enumerate(io.TextIOWrapper(reader, encoding="utf-8")):
        rec = json.loads(line)
        yield i, rec["text"]


def build_splits(docs, tok, fit_tokens: int, eval_tokens: int, max_len: int, min_tokens: int = 16):
    """Assign whole documents in file order: first to F until fit_tokens targets, then E.

    Each document becomes [prefix] + text_ids[: max_len - 1]. Returns lists of
    dicts with doc index, sha256 of the text and the token ids."""
    prefix = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    F, E, nf, ne = [], [], 0, 0
    for idx, text in docs:
        ids = tok(text, add_special_tokens=False)["input_ids"][: max_len - 1]
        if len(ids) < min_tokens:
            continue
        rec = {"doc": idx, "sha256": hashlib.sha256(text.encode()).hexdigest(), "ids": [prefix] + ids}
        if nf < fit_tokens:
            F.append(rec); nf += len(ids)
        elif ne < eval_tokens:
            E.append(rec); ne += len(ids)
        else:
            break
    return F, E


def regular_vocab_mask(toks, vocab_rows: int) -> torch.Tensor:
    """True for ids that are ordinary text tokens of every tokenizer in toks."""
    allowed = torch.zeros(vocab_rows, dtype=torch.bool)
    n = min(min(len(t) for t in toks), vocab_rows)
    allowed[:n] = True
    special = set()
    for tok in toks:
        special |= set(tok.all_special_ids)
        special |= set(getattr(tok, "added_tokens_decoder", {}).keys())
    for i in special:
        if 0 <= i < vocab_rows:
            allowed[i] = False
    return allowed


# ----------------------------------------------------------------------------- caching

@torch.no_grad()
def cache_hidden(model, docs, device, batch_tokens: int = 16384):
    """Run docs through the model; return (H [N,d] bf16, targets [N], pos [N], doc_of [N]) on CPU.

    Row n holds the final normed hidden state at position t, whose target is ids[t+1].
    Position index p = t+1 is the target's position in the document (1 = first text token)."""
    Hs, Ys, Ps, Ds = [], [], [], []
    order = sorted(range(len(docs)), key=lambda i: len(docs[i]["ids"]))
    i = 0
    while i < len(order):
        # order is ascending in length, so the last doc in a group is its longest: grow the
        # group while (size x longest) stays within batch_tokens of padded positions
        group = [order[i]]
        while i + len(group) < len(order) and \
                (len(group) + 1) * len(docs[order[i + len(group)]]["ids"]) <= batch_tokens:
            group.append(order[i + len(group)])
        i += len(group)
        Lmax = len(docs[group[-1]]["ids"])
        x = torch.zeros(len(group), Lmax, dtype=torch.long)
        att = torch.zeros(len(group), Lmax, dtype=torch.long)
        for r, j in enumerate(group):
            ids = docs[j]["ids"]
            x[r, : len(ids)] = torch.tensor(ids)
            att[r, : len(ids)] = 1
        out = model.model(input_ids=x.to(device), attention_mask=att.to(device), use_cache=False)
        h = out.last_hidden_state  # already passed through the final norm
        for r, j in enumerate(group):
            n = len(docs[j]["ids"])
            Hs.append(h[r, : n - 1].to(torch.bfloat16).cpu())
            Ys.append(torch.tensor(docs[j]["ids"][1:n]))
            Ps.append(torch.arange(1, n))
            Ds.append(torch.full((n - 1,), j))
    # kept on CPU while the model is resident; moved to the device after it is freed
    return torch.cat(Hs), torch.cat(Ys), torch.cat(Ps), torch.cat(Ds)


# ----------------------------------------------------------------------------- core math

class Logits:
    """z = h @ W^T restricted to an allowed vocabulary, evaluated in chunks."""

    def __init__(self, H, W, allowed, chunk: int = 4096):
        self.H, self.W, self.chunk = H, W, chunk
        self.neg = torch.zeros(W.shape[0], device=W.device)
        self.neg[~allowed.to(W.device)] = float("-inf")

    def __len__(self):
        return self.H.shape[0]

    def chunks(self, idx=None):
        n = len(self) if idx is None else idx.numel()
        for s in range(0, n, self.chunk):
            sl = slice(s, min(s + self.chunk, n))
            rows = self.H[sl] if idx is None else self.H[idx[sl]]
            yield sl, rows.float() @ self.W.T

    def ce(self, z, y, b=None, logT=None):
        return ce_terms(z, y, b, logT, self.neg)

    def softmax(self, z):
        return torch.softmax(z + self.neg, dim=-1)


def ce_terms(z, y, b=None, logT=None, neg=None):
    """Per-token cross-entropy of softmax(z / T + b + neg) at targets y.

    neg is 0 on allowed ids and -inf elsewhere; it is added after the temperature so
    that d/dlogT never meets -inf * 0."""
    if logT is not None:
        z = z * torch.exp(-logT)
    if b is not None:
        z = z + b
    if neg is not None:
        z = z + neg
    return torch.logsumexp(z, dim=-1) - z.gather(1, y[:, None]).squeeze(1)


@torch.no_grad()
def mean_softmax(L: Logits, idx=None, b=None, logT=None):
    """Mean predictive distribution of softmax(z / T + b) over the rows idx (all rows if None)."""
    acc, n = torch.zeros(L.W.shape[0], device=L.W.device, dtype=torch.float64), 0
    for _, z in L.chunks(idx):
        if logT is not None:
            z = z * torch.exp(-logT)
        if b is not None:
            z = z + b
        acc += L.softmax(z).sum(0).double(); n += z.shape[0]
    return acc / n


@torch.no_grad()
def mean_ce(L: Logits, y, idx, b=None, logT=None):
    return sum(L.ce(z, y[idx[sl]], b, logT).double().sum().item()
               for sl, z in L.chunks(idx)) / idx.numel()


def counts(y, V, idx=None):
    yy = y if idx is None else y[idx]
    return torch.bincount(yy, minlength=V).double()


def closed_form_bias(c, pbar, beta: float):
    """b = log q_s - log pbar with q_s = (c + beta * pbar) / (N + beta)."""
    N = c.sum()
    qs = (c + beta * pbar) / (N + beta)
    b = torch.log(qs) - torch.log(pbar)
    b[~torch.isfinite(b)] = 0.0  # disallowed ids have pbar = 0; their logits are -inf anyway
    return b.float()


def smoothed_target(L: Logits, y, train_idx, beta_frac: float):
    """q_s = (c + beta * pbar0) / (N + beta) on the train rows, beta = beta_frac * N."""
    V = L.W.shape[0]
    c = counts(y, V, train_idx)
    beta = beta_frac * c.sum()
    return (c + beta * mean_softmax(L, train_idx)) / (c.sum() + beta)


@torch.no_grad()
def fit_bias_ipf(L: Logits, y, train_idx, val_idx, target, logT=None, max_iter: int = 20,
                 patience: int = 3, tag: str = "bias", active=None):
    """Iterative proportional fitting of a context-free bias: b <- b + log target - log pbar_b.

    Each token activates exactly one bias coordinate, so this is generalised iterative
    scaling with C = 1: it increases the target-weighted likelihood monotonically, and its
    fixed point is the bias whose train-mean predictive distribution equals the target
    (MDI / dynamic marginals). The first step from b = 0 is the closed-form bias. The
    iterate with the lowest val CE is kept. Deterministic; no line search."""
    allowed = L.neg == 0
    if active is not None:  # only these coordinates may move; the rest stay at 0
        allowed = allowed & active
    logt = torch.where(allowed, target.clamp_min(1e-300).log(), torch.zeros_like(target))
    b = torch.zeros(L.W.shape[0], device=L.W.device)
    best = (mean_ce(L, y, val_idx, b, logT), b.clone(), 0)
    trace = [{"it": 0, "val": best[0]}]
    for it in range(1, max_iter + 1):
        pb = mean_softmax(L, train_idx, b, logT)
        step = logt - pb.log()
        step[~allowed | ~torch.isfinite(step)] = 0.0
        b = b + step.float()
        v = mean_ce(L, y, val_idx, b, logT)
        trace.append({"it": it, "val": v, "max_abs_step": float(step.abs().max())})
        log(f"  fit {tag} it={it} val={v:.6f} max|step|={float(step.abs().max()):.3g}")
        if v < best[0] - 1e-7:
            best = (v, b.clone(), it)
        elif it - best[2] >= patience:
            break
    return best[1], best[0], trace


@torch.no_grad()
def fit_affine(L: Logits, y, train_idx, val_idx, target, max_iter: int = 20,
               lo: float = 0.25, hi: float = 4.0, evals: int = 14, active=None, tag: str = "aff"):
    """Temperature by golden-section search on log T over [lo, hi], scored by the val CE of the
    IPF-fitted bias at that temperature. Returns (b, logT, trace)."""
    g = (math.sqrt(5) - 1) / 2
    a, d = math.log(lo), math.log(hi)
    cache = {}

    def f(x):
        if x not in cache:
            lt = torch.tensor(x, device=L.W.device)
            b, v, _ = fit_bias_ipf(L, y, train_idx, val_idx, target, lt, max_iter,
                                   tag=f"{tag} T={math.exp(x):.3f}", active=active)
            cache[x] = (v, b)
        return cache[x][0]

    x1, x2 = d - g * (d - a), a + g * (d - a)
    f1, f2 = f(x1), f(x2)
    for _ in range(evals - 2):
        if f1 <= f2:
            d, x2, f2 = x2, x1, f1
            x1 = d - g * (d - a); f1 = f(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + g * (d - a); f2 = f(x2)
    xb = min(cache, key=lambda k: cache[k][0])
    trace = [{"T": math.exp(k), "val": v[0]} for k, v in sorted(cache.items())]
    at_edge = min(abs(xb - math.log(lo)), abs(xb - math.log(hi))) < 0.05
    return cache[xb][1], torch.tensor(xb, device=L.W.device), {"grid": trace, "T_at_bracket_edge": at_edge}


@torch.no_grad()
def score(L: Logits, y, pos, doc, n_docs, corrections: dict):
    """Per-document CE sums under each correction, plus per-position-bucket sums."""
    out = {k: torch.zeros(n_docs, dtype=torch.float64, device=y.device) for k in corrections}
    buck = {k: torch.zeros(len(POS_BUCKETS), dtype=torch.float64, device=y.device) for k in corrections}
    ntok = torch.zeros(n_docs, dtype=torch.float64, device=y.device)
    nbuck = torch.zeros(len(POS_BUCKETS), dtype=torch.float64, device=y.device)
    for sl, z in L.chunks():
        yy, pp, dd = y[sl], pos[sl], doc[sl]
        ntok.index_add_(0, dd, torch.ones_like(dd, dtype=torch.float64))
        for bi, (lo, hi) in enumerate(POS_BUCKETS):
            nbuck[bi] += ((pp >= lo) & (pp <= hi)).sum()
        for k, (b, s) in corrections.items():
            ce = L.ce(z, yy, b, s).double()
            out[k].index_add_(0, dd, ce)
            for bi, (lo, hi) in enumerate(POS_BUCKETS):
                m = (pp >= lo) & (pp <= hi)
                buck[k][bi] += ce[m].sum()
    return out, ntok, buck, nbuck


def stats_from_docsums(old: dict, new: dict, ntok, B: int = 2000, seed: int = 0):
    """Point estimates and document-bootstrap 95% CIs of Delta, R's, S_card and S_sym."""
    keys = list(old)
    o = {k: old[k].cpu().numpy() for k in keys}
    n = {k: new[k].cpu().numpy() for k in keys}
    t = ntok.cpu().numpy()

    def point(ix):
        T = t[ix].sum()
        ce_o = {k: o[k][ix].sum() / T for k in keys}
        ce_n = {k: n[k][ix].sum() / T for k in keys}
        d = ce_n["id"] - ce_o["id"]
        r = {"Delta": d}
        for k in keys:
            if k == "id":
                continue
            r[f"R_{k}_new"] = ce_n["id"] - ce_n[k]
            r[f"R_{k}_old"] = ce_o["id"] - ce_o[k]
            r[f"S_card_{k}"] = (ce_n["id"] - ce_n[k]) / d if d != 0 else float("nan")
            r[f"S_sym_{k}"] = 1 - (ce_n[k] - ce_o[k]) / d if d != 0 else float("nan")
        r.update({f"CE_old_{k}": ce_o[k] for k in keys})
        r.update({f"CE_new_{k}": ce_n[k] for k in keys})
        return r

    allix = np.arange(len(t))
    est = point(allix)
    rng = np.random.default_rng(seed)
    boots = [point(rng.integers(0, len(t), len(t))) for _ in range(B)]
    ci = {k: [float(np.nanpercentile([bb[k] for bb in boots], 2.5)),
              float(np.nanpercentile([bb[k] for bb in boots], 97.5))] for k in est}
    return {k: float(v) for k, v in est.items()}, ci


# ----------------------------------------------------------------------------- per model

def run_model(name, rev, F, E, tok, args, device, unrestricted: bool):
    """tok: list of the pair's tokenizers (the vocabulary mask is their intersection)."""
    from transformers import AutoModelForCausalLM
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(name, revision=rev, dtype=torch.bfloat16).to(device).eval()
    W = model.get_output_embeddings().weight.detach().to(torch.bfloat16).float()
    if getattr(model.get_output_embeddings(), "bias", None) is not None:
        raise SystemExit(f"{name} has an output bias; harness assumes none")
    log(f"{name}: loaded; caching FIT")
    HF, yF, pF, dF = cache_hidden(model, F, device, args.batch_tokens)
    log(f"{name}: caching EVAL")
    HE, yE, pE, dE = cache_hidden(model, E, device, args.batch_tokens)
    log(f"{name}: cached {HF.shape[0]} + {HE.shape[0]} positions")
    del model
    torch.cuda.empty_cache() if device.startswith("cuda") else None
    W = W.to(device)
    V = W.shape[0]
    allowed = regular_vocab_mask(tok, V) if not unrestricted else torch.ones(V, dtype=torch.bool)
    # caches are still on the CPU; analyse masks them there and moves only the kept rows
    sums, ntok, buck, nbuck, meta = analyse(HF, yF, pF, dF, HE, yE, pE, dE, W, allowed,
                                            len(F), len(E), args)
    meta.update({"model": name, "revision": rev, "seconds": time.time() - t0})
    return sums, ntok, buck, nbuck, meta


def analyse(HF, yF, pF, dF, HE, yE, pE, dE, W, allowed, nF_docs, nE_docs, args):
    """Everything after caching: fit corrections on F, score E. Model-free, so testable."""
    device = W.device
    V = W.shape[0]
    keepF, keepE = allowed[yF.cpu()], allowed[yE.cpu()]
    HF, yF, pF, dF = (t.cpu()[keepF].to(device) for t in (HF, yF, pF, dF))
    HE, yE, pE, dE = (t.cpu()[keepE].to(device) for t in (HE, yE, pE, dE))
    LF, LE = Logits(HF, W, allowed, args.chunk), Logits(HE, W, allowed, args.chunk)

    # F split for early stopping: last 10% of F documents
    cut = int(0.9 * nF_docs)
    tr_idx = torch.nonzero(dF < cut).squeeze(1)
    va_idx = torch.nonzero(dF >= cut).squeeze(1)

    pbar = mean_softmax(LF)
    c = counts(yF, V)
    corr = {"id": (None, None)}
    for beta in args.betas:
        corr[f"cf_beta{beta:g}"] = (closed_form_bias(c, pbar, beta), None)
    target = smoothed_target(LF, yF, tr_idx, args.beta_frac)
    b_bias, _, tr_b = fit_bias_ipf(LF, yF, tr_idx, va_idx, target, None, args.max_iter)
    b_aff, s_aff, tr_a = fit_affine(LF, yF, tr_idx, va_idx, target, args.max_iter)
    corr["bias"] = (b_bias, None)
    corr["aff"] = (b_aff, s_aff)
    # secondary: bias restricted to the top-K FIT tokens (regularised by support, not by shrinkage)
    ctr = counts(yF, V, tr_idx)
    active = torch.zeros(V, dtype=torch.bool, device=device)
    active[torch.topk(ctr, min(args.topk, V)).indices] = True
    k = f"top{args.topk // 1024}k" if args.topk % 1024 == 0 else f"top{args.topk}"
    b_bk, _, tr_bk = fit_bias_ipf(LF, yF, tr_idx, va_idx, target, None, args.max_iter, tag=f"bias_{k}", active=active)
    b_ak, s_ak, tr_ak = fit_affine(LF, yF, tr_idx, va_idx, target, args.max_iter, active=active, tag=f"aff_{k}")
    corr[f"bias_{k}"] = (b_bk, None)
    corr[f"aff_{k}"] = (b_ak, s_ak)

    log("scoring EVAL")
    sums, ntok, buck, nbuck = score(LE, yE, pE, dE, nE_docs, corr)

    # in-sample Jensen check on E: R_cf(E) >= KL(q_E || pbar_E)
    pbarE = mean_softmax(LE)
    qE = counts(yE, V) / yE.numel()
    m = qE > 0
    kl = float((qE[m] * (qE[m].log() - pbarE[m].log())).sum())
    bE = torch.where(m, qE.clamp_min(1e-300).log() - pbarE.log(), torch.full_like(qE, float("-inf"))).float()
    bE[~allowed.to(device)] = 0.0
    ce_id = ce_ins = 0.0
    for sl, z in LE.chunks():
        ce_id += LE.ce(z, yE[sl]).double().sum().item()
        ce_ins += LE.ce(z, yE[sl], bE).double().sum().item()
    R_in = (ce_id - ce_ins) / yE.numel()

    n_pbar0 = int(((qE > 0) & (pbarE <= 0)).sum())  # fp32 underflow would make the check non-finite
    meta = {"vocab_rows": V, "allowed": int(allowed.sum()), "eval_targets_with_pbar0": n_pbar0,
            "fit_targets": int(yF.numel()), "eval_targets": int(yE.numel()),
            "T_aff": float(torch.exp(s_aff)), "bias_trace": tr_b, "aff_trace": tr_a,
            "T_aff_topk": float(torch.exp(s_ak)), "bias_topk_trace": tr_bk, "aff_topk_trace": tr_ak,
            "jensen_in_sample_E": {"R_cf": R_in, "KL_q_pbar": kl, "holds": R_in >= kl - 1e-9}}
    return sums, ntok, buck, nbuck, meta


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--base_rev", default=None)
    ap.add_argument("--new", required=True, nargs="+")
    ap.add_argument("--new_rev", nargs="+", default=None, help="one per --new; '-' for the default branch")
    ap.add_argument("--data", default=f"https://huggingface.co/datasets/{DCLM_REPO}/resolve/main/{DCLM_SHARD}")
    ap.add_argument("--fit_tokens", type=int, default=1_000_000)
    ap.add_argument("--eval_tokens", type=int, default=1_000_000)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--betas", type=float, nargs="+", default=[1e5, 1e4, 1e6])
    ap.add_argument("--max_iter", type=int, default=20, help="IPF iterations per bias fit")
    ap.add_argument("--beta_frac", type=float, default=0.1)
    ap.add_argument("--topk", type=int, default=4096, help="support of the secondary top-K bias")
    ap.add_argument("--batch_tokens", type=int, default=16384)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--unrestricted", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    revs = a.new_rev or ["-"] * len(a.new)
    if len(revs) != len(a.new):
        raise SystemExit("--new_rev must have one entry per --new")
    revs = [None if r == "-" else r for r in revs]
    torch.backends.cuda.matmul.allow_tf32 = True  # exact for bf16-valued operands
    os.makedirs(a.out, exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.base, revision=a.base_rev)
    toks_new = [AutoTokenizer.from_pretrained(n, revision=r) for n, r in zip(a.new, revs)]
    probe = "The quick brown fox, 1234 jumps; über naïve café."
    for n, t in zip(a.new, toks_new):
        if tok(probe, add_special_tokens=False)["input_ids"] != t(probe, add_special_tokens=False)["input_ids"]:
            raise SystemExit(f"base and {n} tokenizers disagree on plain text")
    F, E = build_splits(iter_dclm_docs(a.data), tok, a.fit_tokens, a.eval_tokens, a.max_len)
    log(f"splits: {len(F)} FIT docs, {len(E)} EVAL docs")
    with open(os.path.join(a.out, "inputs.json"), "w") as fh:
        json.dump({"data": a.data, "fit_docs": [(d["doc"], d["sha256"], len(d["ids"])) for d in F],
                   "eval_docs": [(d["doc"], d["sha256"], len(d["ids"])) for d in E], "args": vars(a)}, fh)
    # one vocabulary mask for the whole group, so the base is fitted once
    all_toks = [tok] + toks_new
    so, nt, bo, nb, mo = run_model(a.base, a.base_rev, F, E, all_toks, a, a.device, a.unrestricted)
    env = {"torch": torch.__version__, "python": sys.version, "platform": platform.platform(),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    outs = []
    for name, rev in zip(a.new, revs):
        sn, _, bn, _, mn = run_model(name, rev, F, E, all_toks, a, a.device, a.unrestricted)
        est, ci = stats_from_docsums(so, sn, nt)
        by_pos = {f"{lo}-{hi}": {"n": float(nb[i]),
                                 "Delta": float((bn["id"][i] - bo["id"][i]) / nb[i]),
                                 "Delta_after_aff": float((bn["aff"][i] - bo["aff"][i]) / nb[i])}
                  for i, (lo, hi) in enumerate(POS_BUCKETS) if nb[i] > 0}
        out = {"pair": [a.base, name], "revisions": [a.base_rev, rev], "estimates": est, "ci95": ci,
               "by_position": by_pos, "old_meta": mo, "new_meta": mn,
               "n_eval_docs": len(E), "n_fit_docs": len(F), "env": env}
        slug = (name.split("/")[-1] + ("@" + rev[:12] if rev else "")).replace("/", "_")
        d = os.path.join(a.out, slug); os.makedirs(d, exist_ok=True)
        np.savez_compressed(os.path.join(d, "docsums.npz"), ntok=nt.cpu().numpy(),
                            **{f"old_{k}": v.cpu().numpy() for k, v in so.items()},
                            **{f"new_{k}": v.cpu().numpy() for k, v in sn.items()})
        with open(os.path.join(d, "result.json"), "w") as fh:
            json.dump(out, fh, indent=1)
        keys = [k for k in est if k.startswith(("Delta", "S_sym_aff", "S_sym_bias", "S_sym_cf_beta100000"))]
        print(slug, json.dumps({k: round(est[k], 4) for k in keys}), flush=True)
        outs.append(out)
    return outs


if __name__ == "__main__":
    main()
