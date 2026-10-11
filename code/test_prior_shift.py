"""Known-answer tests for the prior-shift statistics, on synthetic logits (CPU, about a minute).

Data are sampled from a known "true" model; old and new are constructed so the
share of the loss gap that a context-free correction can recover is known:

  pure bias      new = old + global bias          -> S ~ 1 for bias and aff
  pure temp      new = old * 2                    -> S ~ 1 for aff only, T* ~ 2
  context damage new = old + context-dependent    -> bias shares ~ 0; aff ~ 0.3 (temperature shrinkage)
  proxy          data = true + bias, old = true,  -> S_card inflated; S_sym bias ~ 0
                 new = true + context damage
"""
import argparse
import math

import torch

import prior_shift as ps

V, D, NDOC, LEN = 400, 48, 600, 64


def make(seed=0):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(V, D, generator=g) * 0.5
    H = torch.randn(NDOC * LEN, D, generator=g)
    freq = torch.randn(V, generator=g) * 1.5  # a Zipf-ish unigram prior carried in an extra column
    return g, W, H, freq


def sample(H, W, extra_bias, g):
    z = H @ W.T + extra_bias
    return torch.multinomial(torch.softmax(z, -1), 1, generator=g).squeeze(1)


def aug(H, W, bias):
    """Fold a global bias into an extra hidden coordinate fixed at 1."""
    return torch.cat([H, torch.ones(H.shape[0], 1)], 1), torch.cat([W, bias[:, None]], 1)


def run_case(H_old, W_old, H_new, W_new, y, betas=(1e3,)):
    N = y.numel()
    doc = torch.arange(N) // LEN
    pos = torch.arange(N) % LEN + 1
    nF = NDOC // 2
    F = doc < nF
    args = argparse.Namespace(chunk=4096, betas=list(betas), max_iter=20, beta_frac=0.1, topk=128)
    allowed = torch.ones(V, dtype=torch.bool)
    res = {}
    for role, H, W in (("old", H_old, W_old), ("new", H_new, W_new)):
        res[role] = ps.analyse(H[F], y[F], pos[F], doc[F], H[~F], y[~F], pos[~F], doc[~F] - nF,
                               W, allowed, nF, NDOC - nF, args)
    est, ci = ps.stats_from_docsums(res["old"][0], res["new"][0], res["old"][1], B=200)
    return est, ci, res


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    return cond


def main():
    torch.manual_seed(0)
    g, W, H, freq = make()
    zero = torch.zeros(V)
    y = sample(H, W, freq, g)
    Ho, Wo = aug(H, W, freq)  # old = true model
    ok = True

    # 1. pure global bias shift
    shift = torch.randn(V, generator=g) * 1.0
    Hn, Wn = aug(H, W, freq + shift)
    est, ci, res = run_case(Ho, Wo, Hn, Wn, y)
    print(f"[bias] Delta={est['Delta']:.4f} S_card_bias={est['S_card_bias']:.3f} S_sym_aff={est['S_sym_aff']:.3f} S_card_cf={est['S_card_cf_beta1000']:.3f}")
    ok &= check(est["Delta"] > 0.1, "bias: forgetting is created")
    ok &= check(est["S_card_bias"] > 0.9 and est["S_sym_aff"] > 0.9, "bias: fitted corrections recover >90%")
    ok &= check(est["S_card_cf_beta1000"] > 0.6, "bias: closed form recovers most of it")
    ok &= check(res["new"][4]["jensen_in_sample_E"]["holds"], "bias: Jensen R_cf >= KL on E")

    # 2. pure temperature
    Hn, Wn = aug(H, W / 0.5, freq / 0.5)  # T = 0.5: overconfident new model
    est, ci, res = run_case(Ho, Wo, Hn, Wn, y)
    print(f"[temp] Delta={est['Delta']:.4f} S_card_bias={est['S_card_bias']:.3f} S_card_aff={est['S_card_aff']:.3f} T={res['new'][4]['T_aff']:.3f}")
    ok &= check(est["S_card_aff"] > 0.9, "temp: affine recovers >90%")
    ok &= check(abs(res["new"][4]["T_aff"] - 2.0) < 0.1, "temp: fitted T ~ 2 undoes the doubled logits")
    ok &= check(est["S_card_bias"] < est["S_card_aff"] - 0.2, "temp: bias alone recovers clearly less")

    # 3. context-dependent damage only
    Wd = W + torch.randn(V, D, generator=g) * 0.25
    Hn, Wn = aug(H, Wd, freq)
    est, ci, res = run_case(Ho, Wo, Hn, Wn, y)
    print(f"[ctx] Delta={est['Delta']:.4f} S_sym_aff={est['S_sym_aff']:.3f} S_sym_bias={est['S_sym_bias']:.3f} S_sym_cf={est['S_sym_cf_beta1000']:.3f}")
    ok &= check(est["Delta"] > 0.05, "ctx: forgetting is created")
    ok &= check(est["S_sym_bias"] < 0.1 and est["S_sym_cf_beta1000"] < 0.1, "ctx: bias shares small")
    # Random context damage inflates the logit scale, and a
    # temperature recovers part of that loss (shrinkage). This is the noise floor of S_aff
    # discussed in README.md.
    ok &= check(0.15 < est["S_sym_aff"] < 0.45, "ctx: affine share sits at the temperature noise floor (~0.3)")

    # 4. proxy mismatch: data drawn from true+bias; old = true (miscalibrated on proxy); new = ctx-damaged
    y2 = sample(H, W, freq + shift * 0.7, g)
    est, ci, res = run_case(Ho, Wo, Hn, Wn, y2)
    print(f"[proxy] Delta={est['Delta']:.4f} S_card_aff={est['S_card_aff']:.3f} S_sym_aff={est['S_sym_aff']:.3f} S_card_bias={est['S_card_bias']:.3f} S_sym_bias={est['S_sym_bias']:.3f}")
    ok &= check(est["R_aff_old"] > 0.05, "proxy: base model itself gains from the correction")
    ok &= check(est["S_card_aff"] > est["S_sym_aff"] + 0.1, "proxy: card statistic is inflated relative to symmetric")
    ok &= check(est["S_sym_bias"] < 0.15, "proxy: symmetric bias share stays small")

    # 5. no change: Delta == 0 exactly (ratios are then undefined; summarize.py marks such a pair invalid)
    est, ci, res = run_case(Ho, Wo, Ho, Wo, y)
    ok &= check(abs(est["Delta"]) < 1e-9, "identity: Delta == 0 exactly")
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
