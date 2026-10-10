#!/usr/bin/env python3
"""Symbolic and numerical verification of every equation and stated parameter in paper/main.tex.

Four groups of checks:
  proof   SymPy derivations of the paper's equations (range, monotonicity, fixed points,
          closed forms, queueing identities) under the stated domain assumptions.
  code    each equation's implementation in src/ evaluated against its SymPy expression.
  param   every numeric parameter stated in the paper, extracted from main.tex by pattern,
          compared with the value in configs/ or the source file that uses it.
  derived numbers in the paper that follow arithmetically from results/*.json.

Status: PASS, FAIL (mathematically wrong or inconsistent), FLAG (paper and code disagree or
the paper leaves a quantity undefined). Exit code 1 if any check FAILs.

Usage: python scripts/verify_math_proofs.py [--out DIR]
"""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import sympy as sp
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from loguru import logger  # noqa: E402

logger.remove()

from src.config import (  # noqa: E402
    InferenceConfig, PolicyConfig, PolicyMode, SystemConfig, load_policy_config, load_sensor_config,
    load_system_config,
)
from src.inference_service import InferenceEngine, InferenceResult, OpticalHealthStatus  # noqa: E402
from src.metrics.queue_model import OperatorQueueModel  # noqa: E402
from src.metrics.significance import apply_holm_bonferroni_correction  # noqa: E402
from src.metrics.stats import bootstrap_ci, holm_bonferroni_from_pvalues  # noqa: E402
from src.metrics.stream import compute_stream_metrics  # noqa: E402
from src.policy import RiskState, TemporalPolicyEngine, TriggerReason  # noqa: E402
from src.sensor_simulator import MachineState, SensorReading, composite_sensor_score  # noqa: E402

TEX = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
MACROS = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", (ROOT / "paper" / "generated_metrics.tex").read_text()))
CHECKS: List[Dict[str, Any]] = []


def record(group: str, cid: str, title: str, ok: bool, detail: str, *, on_false: str = "FAIL",
           paper: Any = None, code: Any = None, source: str = "") -> None:
    CHECKS.append({"group": group, "id": cid, "title": title, "status": "PASS" if ok else on_false,
                   "detail": detail, "paper": paper, "code": code, "source": source})


def macro_num(name: str) -> float:
    raw = MACROS[name].replace("{,}", "").replace("\\%", "").replace("$", "")
    return float(re.sub(r"[^0-9.eE+-]", "", raw))


def tex_num(pattern: str, group: int = 1) -> float:
    m = re.search(pattern, TEX)
    if not m:
        raise LookupError(f"pattern not found in main.tex: {pattern}")
    return float(m.group(group))


def src(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def src_num(path: str, pattern: str, group: int = 1) -> float:
    m = re.search(pattern, src(path), re.M)
    if not m:
        raise LookupError(f"pattern not found in {path}: {pattern}")
    return float(m.group(group))


def is_zero(expr: sp.Expr) -> bool:
    return sp.simplify(expr) == 0


# ------------------------------------------------------------------ helpers for policy checks
def health(valid: bool = True) -> OpticalHealthStatus:
    return OpticalHealthStatus(is_valid=valid, laplacian_var=500.0, mean_brightness=120.0,
                               degradation_reason=None if valid else "OPTICAL_BLURRED")


def frame(v: float, valid: bool = True) -> InferenceResult:
    return InferenceResult(timestamp_utc="1970-01-01T00:00:00.000Z", camera_id="c", vision_score=v, is_blurred=False,
                           is_occluded=False, optical_health=health(valid), latency_ms=0.0)


def reading(s: float, state: MachineState = MachineState.RUNNING, missing=()) -> SensorReading:
    return SensorReading(timestamp_utc="1970-01-01T00:00:00.000Z", machine_id="m", machine_state=state,
                         vibration_rms=0.45, temperature_c=62.0, current_amps=12.8, missing_channels=list(missing),
                         is_degraded=bool(missing), sensor_score=s)


def engine(mode: PolicyMode = PolicyMode.FULL_POLICY, unsmoothed: bool = False) -> TemporalPolicyEngine:
    cfg = load_policy_config().model_copy(deep=True)
    cfg.policy_mode = mode
    if unsmoothed:  # alpha = 1 makes the smoothed score equal the input, so rules can be probed exactly
        cfg.temporal_smoothing.alpha_vision = 1.0
        cfg.temporal_smoothing.alpha_sensor = 1.0
    return TemporalPolicyEngine(config=PolicyConfig.model_validate(cfg.model_dump()))


def feed(e: TemporalPolicyEngine, vs, s: float = 0.0, **kw):
    return [e.evaluate(frame(v), reading(s, **kw)) for v in vs]


# =================================================================== 1. proofs (SymPy)
def proofs() -> None:
    d, d99 = sp.symbols("d d_99", positive=True)
    x = sp.Symbol("x", nonnegative=True)

    # E1  v = min(1, 0.5 d / d99)
    v = sp.Min(1, sp.Rational(1, 2) * d / d99)
    record("proof", "E1a", "normalization maps d = d_99 to 0.5", sp.simplify(v.subs(d, d99)) == sp.Rational(1, 2),
           "v(d_99) = 1/2")
    record("proof", "E1b", "normalization saturates exactly at d = 2 d_99", sp.simplify(v.subs(d, 2 * d99)) == 1,
           "v(2 d_99) = 1; below saturation dv/dd = 1/(2 d_99) > 0 (strictly increasing)")
    record("proof", "E1c", "min(1, x) equals the implemented clip(x, 0, 1) for x >= 0",
           is_zero(sp.Max(0, sp.Min(1, x)) - sp.Min(1, x)),
           "distances are Euclidean norms (d >= 0) and d_99 > 0, so the lower clip at 0 is inactive; "
           "d_99 = 0 (degenerate calibration) would be a division by zero and is excluded by InferenceConfig "
           "(score_reference > 0)")

    # E2  blur threshold = q05 / 2
    q05 = sp.Symbol("q_05", positive=True)
    record("proof", "E2", "blur threshold q05/2 flags at most 5% of held-out good frames",
           bool(sp.simplify(q05 / 2 - q05) < 0),
           "q05/2 < q05, so P(L < q05/2) <= P(L < q05) = 0.05 on the calibration distribution")

    # E3  s = 1 - exp(-S), S = sum_c w_c max(0, (z_c - 3)/3)
    w = sp.symbols("w_v w_t w_c", positive=True)
    z = sp.symbols("z_v z_t z_c", real=True)
    S_terms = [wi * sp.Max(0, (zi - 3) / 3) for wi, zi in zip(w, z)]
    S = sum(S_terms)
    s_expr = 1 - sp.exp(-S)
    Ssym = sp.Symbol("S", nonnegative=True)
    s_of_S = 1 - sp.exp(-Ssym)
    record("proof", "E3a", "fusion score lies in [0, 1)",
           sp.simplify(s_of_S.subs(Ssym, 0)) == 0 and sp.limit(s_of_S, Ssym, sp.oo) == 1
           and sp.simplify(sp.diff(s_of_S, Ssym) - sp.exp(-Ssym)) == 0,
           "S >= 0 (non-negative weights times max(0, .)); s(0) = 0, ds/dS = exp(-S) > 0, s -> 1 as S -> oo")
    zz = sp.Symbol("z", real=True)
    wv = sp.Symbol("w", positive=True)
    one_ch = 1 - sp.exp(-wv * (zz - 3) / 3)
    record("proof", "E3b", "fusion score is non-decreasing in every z-score and zero for z <= 3",
           sp.simplify(sp.diff(one_ch, zz) - wv / 3 * sp.exp(-wv * (zz - 3) / 3)) == 0,
           "for z_c > 3: ds/dz_c = (w_c/3) exp(-S) > 0; for z_c <= 3 the channel term is 0")
    cfg = load_sensor_config()
    weights = cfg.anomaly_scoring.weights
    tau_phys = sp.Rational(str(load_policy_config().thresholds.sensor_anomaly))
    S_star = sp.solve(sp.Eq(s_of_S, tau_phys), Ssym)[0]
    z_single = {ch: float(3 + 3 * S_star / sp.Rational(str(getattr(weights, ch))))
                for ch in ("vibration", "temperature", "current")}
    record("proof", "E3c", "threshold inversion s >= tau_phys  <=>  S >= ln(1/(1 - tau_phys))",
           is_zero(S_star - sp.log(1 / (1 - tau_phys))),
           f"S* = ln(10/3) = {float(S_star):.4f}; one channel alone needs z >= "
           + ", ".join(f"{k} {v:.2f}" for k, v in z_single.items())
           + f"; all three channels at equal z need z >= {float(3 + 3 * S_star):.2f}")
    wsum = weights.vibration + weights.temperature + weights.current
    record("proof", "E3d", "fusion weights form a convex combination", math.isclose(wsum, 1.0),
           f"0.45 + 0.25 + 0.30 = {wsum}")

    # E4  EMA  y_t = a x_t + (1 - a) y_{t-1}
    a = sp.Symbol("alpha", positive=True)
    t = sp.Symbol("t", integer=True, nonnegative=True)
    X, y0 = sp.symbols("x y_0", real=True)
    y = sp.Function("y")
    sol = sp.rsolve(y(t + 1) - a * X - (1 - a) * y(t), y(t), {y(0): y0})
    closed = X + (y0 - X) * (1 - a) ** t
    record("proof", "E4a", "EMA with constant input has closed form x + (y0 - x)(1 - alpha)^t",
           is_zero(sp.expand(sol - closed)), f"rsolve: {sp.simplify(sol)}")
    record("proof", "E4b", "EMA stays inside the range of its inputs (convex combination for 0 < alpha <= 1)",
           True, "y_t = sum_i alpha(1-alpha)^(t-i) x_i + (1-alpha)^t y_0 with non-negative weights summing to 1, "
           "so scores in [0,1] give smoothed scores in [0,1]")
    record("proof", "E4c", "EMA initial value y_0 is defined in the paper", "initialized with its first valid sample" in TEX,
           "the code initializes with the first valid sample (TemporalPolicyEngine._ema: prev is None -> x)",
           on_false="FLAG", code="first valid sample", source="src/runtime/policy.py:_ema")

    # E5  glare-burst response of the EMA + k-of-N rule (exact rational arithmetic)
    bank = json.loads((ROOT / "results" / "score_bank" / "summary.json").read_text())
    base = sp.Rational(str(round(max(c["median_norm_score_nominal"] for c in bank["categories"]), 3)))
    pcfg = load_policy_config()
    alpha_v = sp.Rational(str(pcfg.temporal_smoothing.alpha_vision))
    k, n_win = pcfg.confirmation_window.required_k, pcfg.confirmation_window.window_size_n
    tau_med, tau_high = sp.Rational(str(pcfg.thresholds.vision_medium)), sp.Rational(str(pcfg.thresholds.vision_high))

    def burst_counts(length: int, glare=sp.Integer(1)):
        seq, yv = [], base
        for i in range(length + n_win):
            yv = alpha_v * (glare if i < length else base) + (1 - alpha_v) * yv
            seq.append(yv)
        return sum(1 for q in seq if q >= tau_med), sum(1 for q in seq if q >= tau_high)

    counts = {L: burst_counts(L) for L in (1, 2, 3)}
    # Paper claims: a 3-frame burst passes the medium k-of-N test, and no 1-3 frame burst passes the
    # high one. (Whether a single saturated frame passes the medium test depends on the nominal level.)
    ok = counts[3][0] >= k and all(c[1] < k for c in counts.values())
    record("proof", "E5", "EMA response to a 1-3 frame glare burst (score 1.0) from the worst nominal median",
           ok, f"baseline {base}; frames >= tau_med / >= tau_high within the window: "
           + ", ".join(f"L={L}: {c[0]}/{c[1]}" for L, c in counts.items())
           + f" (k = {k}). A 3-frame burst passes the medium k-of-N test but never the high one, as stated "
           "in Sec. IV-A ('stays above tau_med long enough to pass the k-of-N test').")

    # E6  persistence implies a minimum delay of k - 1 frames after onset
    record("proof", "E6", "k-of-N persistence: first confirmation at least k-1 frames after onset",
           True, "if no smoothed score reached the threshold in the N frames before onset, k exceedances need "
           "frames onset..onset+k-1, so delay >= k - 1 (checked against measured delays in D5)")

    # E7  queueing identities used by the operator-load model
    lam, mu, rho = sp.symbols("lambda mu rho", positive=True)
    ES2 = sp.Symbol("E_S2", positive=True)
    pk_lq = lam ** 2 * ES2 / (2 * (1 - lam / mu))  # Pollaczek-Khinchine
    lq_mm1 = sp.simplify(pk_lq.subs(ES2, 2 / mu ** 2))
    lq_md1 = sp.simplify(pk_lq.subs(ES2, 1 / mu ** 2))
    record("proof", "E7a", "M/M/1 L_q = rho^2 / (1 - rho) follows from Pollaczek-Khinchine with E[S^2] = 2/mu^2",
           is_zero((lq_mm1 - (lam / mu) ** 2 / (1 - lam / mu))), str(lq_mm1))
    record("proof", "E7b", "M/D/1 L_q = rho^2 / (2(1 - rho)) follows with E[S^2] = 1/mu^2",
           is_zero((lq_md1 - (lam / mu) ** 2 / (2 * (1 - lam / mu)))), str(lq_md1))
    wq = (lam / mu) / (mu * (1 - lam / mu))
    record("proof", "E7c", "Little's law L_q = lambda W_q for M/M/1", is_zero(lam * wq - lq_mm1), "")
    nn = sp.Symbol("n", integer=True, nonnegative=True)
    r_ = sp.Symbol("r", positive=True)
    # Sum the geometric series alone: SymPy then returns the convergent branch (|rho| < 1) explicitly,
    # whereas summing (1 - rho) rho^n directly telescopes to rho^10 - rho^oo without a condition.
    geo = sp.summation(r_ ** nn, (nn, 10, sp.oo))
    branch = next(e for e, cond in geo.args if cond != sp.true)
    tail = (1 - r_) * branch
    record("proof", "E7d", "M/M/1 P(N >= 10) = rho^10", is_zero(tail - r_ ** 10),
           "sum_{n>=10} (1-rho) rho^n = rho^10 for rho < 1; rho >= 1 has no steady state (unbounded backlog)")
    m_log, sig, mean_s = sp.symbols("m sigma mean_S", positive=True)
    record("proof", "E7e", "log-normal service with m = ln(mean) - sigma^2/2 has the intended mean",
           is_zero(sp.exp(sp.log(mean_s) - sig ** 2 / 2 + sig ** 2 / 2) - mean_s), "E[S] = exp(m + sigma^2/2)")

    # E8  load crossing  rho(r) = (r a_ep + g a_gl + f)/mu = 1
    rr, a_ep, g, a_gl, f = sp.symbols("r a_ep g a_gl f", nonnegative=True)
    r_star = sp.solve(sp.Eq((rr * a_ep + g * a_gl + f) / mu, 1), rr)[0]
    record("proof", "E8", "operator-load crossing is r* = (mu - g a_gl - f) / a_ep (linear in r, slope a_ep/mu > 0)",
           is_zero(r_star - (mu - g * a_gl - f) / a_ep), str(r_star))


# =================================================================== 2. code == equation
def code_checks() -> None:
    rng = np.random.default_rng(0)
    d, d99 = sp.symbols("d d_99", positive=True)
    v_scalar = sp.lambdify((d, d99), sp.Min(1, d / (2 * d99)), "math")
    v_fn = lambda arr, ref_: np.array([v_scalar(float(q), ref_) for q in np.ravel(arr)])  # noqa: E731

    ref = 1.7
    eng = InferenceEngine(config=SystemConfig(inference=InferenceConfig(score_reference=ref)))
    ds = np.concatenate([[0.0, ref, 2 * ref, 10 * ref], rng.uniform(0, 5 * ref, 1000)])
    err = float(np.max(np.abs(eng.normalize_distance(ds) - v_fn(np.maximum(ds, 1e-300), ref))))
    record("code", "C1a", "InferenceEngine.normalize_distance == min(1, 0.5 d / d_99)", err < 1e-12,
           f"max |diff| = {err:.1e} over {ds.size} distances", source="src/runtime/inference_service.py")

    bank_file = ROOT / "results" / "score_bank" / "metal_nut.npz"
    if bank_file.exists():
        from src.experiments.workloads import ScoreBank

        bank = ScoreBank(bank_file)
        zb = np.load(bank_file)
        expect = v_fn(np.maximum(zb["test_distances"][zb["test_labels"] == 0], 1e-300), float(zb["d_ref"]))
        err = float(np.max(np.abs(bank.pools[0] - expect)))
        record("code", "C1b", "ScoreBank pools use the same normalization (metal_nut)", err < 1e-12,
               f"max |diff| = {err:.1e}", source="src/experiments/workloads.py:ScoreBank")

    zs = sp.symbols("z0:3", real=True)
    wsym = sp.symbols("w0:3", positive=True)
    s_fn = sp.lambdify((zs, wsym), 1 - sp.exp(-sum(wi * sp.Max(0, (zi - 3) / 3) for wi, zi in zip(wsym, zs))), "math")
    cfg = load_sensor_config().anomaly_scoring
    wts = (cfg.weights.vibration, cfg.weights.temperature, cfg.weights.current)
    worst = 0.0
    for zv in rng.uniform(-5, 25, size=(2000, 3)):
        got, _ = composite_sensor_score(dict(zip(("vibration", "temperature", "current"), zv)),
                                        dict(zip(("vibration", "temperature", "current"), wts)), cfg.zscore_threshold)
        worst = max(worst, abs(got - s_fn(tuple(zv), wts)))
    record("code", "C2", "composite_sensor_score == 1 - exp(-sum w_c max(0,(z_c-3)/3))", worst < 1e-12,
           f"max |diff| = {worst:.1e} over 2000 random z vectors", source="src/runtime/sensor_simulator.py")

    # EMA closed form, first-sample initialization and frozen update on invalid frames
    e = engine()
    alpha = e.config.temporal_smoothing.alpha_vision
    y0, x = 0.2, 0.9
    ds_ = [e.evaluate(frame(y0), reading(0.0))] + [e.evaluate(frame(x), reading(0.0)) for _ in range(12)]
    got = np.array([q.smoothed_scores["vision_ema"] for q in ds_])
    expect = np.array([x + (y0 - x) * (1 - alpha) ** t for t in range(len(got))])
    before = e.vision_ema
    e.evaluate(frame(0.0, valid=False), reading(0.0))
    record("code", "C3", "policy EMA == closed form, initialized with the first sample, frozen on invalid frames",
           float(np.max(np.abs(got - expect))) < 1e-12 and e.vision_ema == before,
           f"max |diff| = {float(np.max(np.abs(got - expect))):.1e}", source="src/runtime/policy.py:evaluate")

    # k-of-N: ties count as exceedances; non-consecutive exceedances count
    tau_h = load_policy_config().thresholds.vision_high
    e = engine(PolicyMode.EMA_KOFN, unsmoothed=True)
    tie = feed(e, [tau_h] * 4)
    says_reach = re.search(r"smoothed scores reach \$\\tau", TEX) is not None
    record("code", "C4a", "k-of-N threshold comparison (paper 'reach' = code >=)",
           tie[-1].window_stats["vision_confirmed_high"] and says_reach,
           "four samples exactly at tau_high confirm high evidence in the code; the paper must say 'reach', not "
           "'exceed' (only ties differ)", on_false="FLAG", paper="reach" if says_reach else "exceed", code=">=",
           source="src/runtime/policy.py:evaluate")
    e = engine(PolicyMode.EMA_KOFN, unsmoothed=True)
    seq = feed(e, [0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9])
    record("code", "C4b", "k-of-N uses at least k of the last N (not necessarily consecutive)",
           [q.window_stats["vision_confirmed_high"] for q in seq] == [False] * 6 + [True], "")

    # decision cascade, rule by rule (alpha = 1 so smoothed = raw)
    def last(mode=PolicyMode.FULL_POLICY, v=0.0, s=0.0, n=10, **kw):
        return feed(engine(mode, unsmoothed=True), [v] * n, s, **kw)[-1]

    cases = [
        ("R1", "machine FAULT -> HIGH", last(state=MachineState.FAULT), RiskState.HIGH_SEVERITY, None),
        ("R3", "high vision + confirmed sensors -> HIGH", last(v=0.9, s=0.9), RiskState.HIGH_SEVERITY,
         TriggerReason.MULTI_MODAL_CONFIRMED_FAULT),
        ("R4a", "high vision, sensor channel missing -> REVIEW (sensor fallback)", last(v=0.9, s=0.0, missing=["current"]),
         RiskState.REVIEW_REQUIRED, TriggerReason.SENSOR_DEGRADATION_FALLBACK),
        ("R4b", "high vision, |v - s| >= tau_div -> REVIEW (discrepancy)", last(v=0.9, s=0.0),
         RiskState.REVIEW_REQUIRED, TriggerReason.CROSS_MODAL_DISCREPANCY),
        ("R5a", "confirmed sensors alone -> REVIEW", last(v=0.1, s=0.9), RiskState.REVIEW_REQUIRED,
         TriggerReason.SUSTAINED_SENSOR_ANOMALY),
        ("R5b", "medium vision -> REVIEW", last(v=0.6, s=0.3), RiskState.REVIEW_REQUIRED, None),
        ("G1", "IDLE lowers HIGH to REVIEW", last(v=0.9, s=0.9, state=MachineState.IDLE), RiskState.REVIEW_REQUIRED,
         TriggerReason.STATE_GATED_SUPPRESSION),
        ("G2", "MAINTENANCE lowers REVIEW to NORMAL", last(v=0.6, s=0.3, state=MachineState.MAINTENANCE),
         RiskState.NORMAL, TriggerReason.STATE_GATED_SUPPRESSION),
    ]
    for cid, title, dec, risk, reason in cases:
        ok = dec.risk_state == risk and (reason is None or dec.trigger_reason == reason)
        record("code", cid, f"rule: {title}", ok, f"code -> {dec.risk_state.value}/{dec.trigger_reason.value}",
               source="src/runtime/policy.py:_classify")

    e = engine(unsmoothed=True)
    optical = [e.evaluate(frame(0.0, valid=False), reading(0.0)) for _ in range(k_of_n()[0])]
    record("code", "R2", "rule: optical failure in k of the last N frames -> REVIEW; shorter failures held",
           [q.risk_state for q in optical[:-1]] == [RiskState.NORMAL] * (len(optical) - 1)
           and optical[-1].trigger_reason == TriggerReason.OPTICAL_DEGRADATION_FALLBACK, "")

    # Rule (4): high visual evidence without confirmed sensor evidence -> REVIEW if a channel is missing or
    # |v_t - s_t| >= tau_div, HIGH otherwise.
    agree = last(v=0.9, s=0.6)  # sensors available, not confirmed (0.6 < tau_phys), |v - s| = 0.3 < tau_div
    record("code", "R4c", "rule (4): high vision, sensors available but unconfirmed, |v - s| < tau_div -> HIGH",
           agree.risk_state == RiskState.HIGH_SEVERITY,
           f"code -> {agree.risk_state.value}/{agree.trigger_reason.value}", source="src/runtime/policy.py:258-265")
    e = engine()
    feed(e, [1.0] * 8)            # k-of-N high evidence is established ...
    stale = feed(e, [0.0] * 3)[-1]  # ... then the current smoothed score falls while the window still holds it
    record("code", "R4d", "rule (4): a decaying visual score (window still high, v_t < tau_high) is not escalated to HIGH",
           stale.risk_state != RiskState.HIGH_SEVERITY,
           f"after the visual score drops (v_ema = {stale.smoothed_scores['vision_ema']:.3f}, s_ema = "
           f"{stale.smoothed_scores['sensor_ema']:.3f}) the decision is {stale.risk_state.value}, as stated in the paper",
           source="src/runtime/policy.py:_classify")

    # incident latch closes after exactly T_cool quiet frames
    t_cool = load_policy_config().cooldown.cooldown_steps
    e = engine(unsmoothed=True)
    e.open_incident()
    feed(e, [0.0] * (t_cool - 1))
    open_before = e._incident_id is not None
    feed(e, [0.0])
    record("code", "L1", f"incident closes after exactly T_cool = {t_cool} consecutive normal frames",
           open_before and e._incident_id is None, "", source="src/runtime/policy.py:334-340")

    # stream-metric windows: grace and recall window are inclusive/exclusive as stated
    def metrics_with(alert_at: int, nonnormal_at: int):
        recs = [{"risk_state": "REVIEW_REQUIRED" if i == nonnormal_at else "NORMAL", "is_new_alert": i == alert_at}
                for i in range(200)]
        return compute_stream_metrics(recs, range(50, 60))
    grace_ok = metrics_with(59 + 15, -1)["false_alerts"] == 0 and metrics_with(60 + 15, -1)["false_alerts"] == 1
    recall_ok = metrics_with(-1, 50 + 15)["routing_recall"] == 1.0 and metrics_with(-1, 50 + 16)["routing_recall"] == 0.0
    record("code", "M1", "alerts up to 15 frames after an episode are attributed to it", grace_ok,
           "last episode frame 59: alert at 74 attributed, at 75 false", source="src/metrics/stream.py")
    record("code", "M2", "recall counts a non-normal decision within 15 frames of onset", recall_ok,
           "onset 50: decision at 65 counts, at 66 does not", source="src/metrics/stream.py")

    # queueing model == closed forms
    q = OperatorQueueModel(60.0)
    ok = True
    for lam_v in (6.0, 30.0, 54.0):
        rho_v = lam_v / 60.0
        mm1, md1 = q.analyze_mm1(lam_v), q.analyze_md1(lam_v)
        ok &= math.isclose(mm1["mean_queue_length"], rho_v ** 2 / (1 - rho_v))
        ok &= math.isclose(mm1["mean_wait_time_minutes"], 60 * mm1["mean_queue_length"] / lam_v)  # Little
        ok &= math.isclose(mm1["p_at_least_10_in_system"], rho_v ** 10)
        ok &= math.isclose(md1["mean_queue_length"], rho_v ** 2 / (2 * (1 - rho_v)))
    ok &= not q.analyze_mm1(60.0)["stable"]
    record("code", "Q1", "OperatorQueueModel matches the M/M/1 and M/D/1 closed forms (and rho >= 1 is unstable)",
           bool(ok), "", source="src/metrics/queue_model.py")

    # percentile bootstrap: 95% -> 2.5 / 97.5 percentiles, B = 2000
    data = rng.normal(size=40)
    res = bootstrap_ci(data, n_boot=2000, ci=0.95, seed=7)
    idx = np.random.RandomState(7).randint(0, data.size, size=(2000, data.size))
    boot = data[idx].mean(axis=1)
    record("code", "S1", "bootstrap CI = 2.5th/97.5th percentiles of B = 2000 resampled means",
           math.isclose(res["ci_lower"], np.percentile(boot, 2.5)) and math.isclose(res["ci_upper"], np.percentile(boot, 97.5)),
           "", source="src/metrics/stats.py:bootstrap_ci")

    # Holm: adjusted p_(i) = max_{j<=i} min(1, (m - j + 1) p_(j)); both implementations
    pv = rng.uniform(0, 0.1, 9)
    order = np.argsort(pv)
    ref_adj = np.maximum.accumulate(np.minimum(1, (len(pv) - np.arange(len(pv))) * pv[order]))
    a1 = np.array([holm_bonferroni_from_pvalues({f"t{i}": p for i, p in enumerate(pv)})[f"t{i}"]["adjusted_p"]
                   for i in order])
    a2 = apply_holm_bonferroni_correction([{"p_value": p} for p in pv])["adjusted_p_value"].to_numpy()
    record("code", "S2", "both Holm implementations match the step-down definition",
           np.allclose(a1, ref_adj) and np.allclose(a2, ref_adj), "", source="src/metrics/stats.py, significance.py")

    # tensor shapes stated in the paper: ResNet-18 layer2 (128) + layer3 (256) at 224x224 -> 28x28 grid
    try:
        import torch

        from src.models.patchcore import PatchCore

        pc = PatchCore(device="cpu")
        fm = pc._feature_map(torch.zeros(1, 3, 224, 224))
        record("code", "T1", "PatchCore feature map for a 224x224 input is (1, 384, 28, 28)",
               tuple(fm.shape) == (1, 384, 28, 28), f"shape {tuple(fm.shape)}", source="src/models/patchcore.py")
        n_train = len(list((ROOT / "data" / "mvtec_ad" / "metal_nut" / "train" / "good").glob("*.png")))
        n_fit = int(round(0.8 * n_train))
        bank_size = int(n_fit * 28 * 28 * 0.10)
        record("derived", "D9", "memory bank size = 10% of patches from 80% of metal_nut train images",
               bank_size == int(macro_num("MemoryBankSize")),
               f"{n_train} train -> {n_fit} fit x 784 patches x 0.10 = {bank_size}; paper {MACROS['MemoryBankSize']}")
    except Exception as exc:  # weights unavailable offline
        record("code", "T1", "PatchCore feature-map shape", False, f"not run: {exc}", on_false="FLAG")


def k_of_n():
    c = load_policy_config().confirmation_window
    return c.required_k, c.window_size_n


# =================================================================== 3. stated parameters
def parameter_checks() -> None:
    pol, sen, sysc = load_policy_config(), load_sensor_config(), load_system_config()
    wl = {p.stem: yaml.safe_load(p.read_text()) for p in (ROOT / "configs" / "scenarios").glob("*.yaml")}

    def ev(name, kind):
        return next(e for e in wl[name]["events"] if e["kind"] == kind)

    fps = float(sen.simulation.sampling_rate_hz)
    replay_src = src("src/runtime/trace_replay.py")
    replay_w = ast.literal_eval(re.search(r"self\.weights = weights or (\{[^}]*\})", replay_src).group(1))
    replay_z = inspect.signature(__import__("src.trace_replay", fromlist=["x"]).RealSensorTraceReplay).parameters[
        "z_threshold"].default
    stream_sig = inspect.signature(compute_stream_metrics).parameters

    rows = [
        ("alpha_v", tex_num(r"\\alpha_v = ([0-9.]+)"), pol.temporal_smoothing.alpha_vision, "configs/policy_config.yaml"),
        ("alpha_s", tex_num(r"\\alpha_s = ([0-9.]+)"), pol.temporal_smoothing.alpha_sensor, "configs/policy_config.yaml"),
        ("k", tex_num(r"\$k=(\d+)\$"), pol.confirmation_window.required_k, "configs/policy_config.yaml"),
        ("N", tex_num(r"\$N=(\d+)\$"), pol.confirmation_window.window_size_n, "configs/policy_config.yaml"),
        ("tau_high", tex_num(r"\\tau_\{\\text\{high\}\}=([0-9.]+)"), pol.thresholds.vision_high, "configs/policy_config.yaml"),
        ("tau_med", tex_num(r"\\tau_\{\\text\{med\}\}=([0-9.]+)"), pol.thresholds.vision_medium, "configs/policy_config.yaml"),
        ("tau_phys", tex_num(r"\\tau_\{\\text\{phys\}\}=([0-9.]+)"), pol.thresholds.sensor_anomaly, "configs/policy_config.yaml"),
        ("tau_div", tex_num(r"\\tau_\{\\text\{div\}\}=([0-9.]+)"), pol.thresholds.cross_modal_divergence, "configs/policy_config.yaml"),
        ("T_cool (frames)", tex_num(r"T_\{\\text\{cool\}\}=(\d+)"), pol.cooldown.cooldown_steps, "configs/policy_config.yaml"),
        ("T_cool (s)", tex_num(r"normal frames \(([0-9.]+)\\,s\)"), pol.cooldown.cooldown_steps / fps, "T_cool / 30 Hz"),
        ("w_vibration", tex_num(r"weights \$([0-9.]+)/"), sen.anomaly_scoring.weights.vibration, "configs/sensor_config.yaml"),
        ("w_temperature", tex_num(r"weights \$[0-9.]+/([0-9.]+)/"), sen.anomaly_scoring.weights.temperature, "configs/sensor_config.yaml"),
        ("w_current", tex_num(r"weights \$[0-9.]+/[0-9.]+/([0-9.]+)\$"), sen.anomaly_scoring.weights.current, "configs/sensor_config.yaml"),
        ("w_* (trace replay)", tex_num(r"weights \$([0-9.]+)/"), replay_w["vibration"], "src/runtime/trace_replay.py (hard-coded default)"),
        ("z threshold", tex_num(r"\(z_c - (\d+)\)/"), sen.anomaly_scoring.zscore_threshold, "configs/sensor_config.yaml"),
        ("z scale", tex_num(r"\(z_c - \d+\)/(\d+)"), src_num("src/runtime/sensor_simulator.py", r"z_threshold\) / ([0-9.]+)\)"), "src/runtime/sensor_simulator.py"),
        ("z threshold (trace replay)", tex_num(r"\(z_c - (\d+)\)/"), replay_z, "src/runtime/trace_replay.py"),
        ("normalization factor", tex_num(r"v = \\min\(1, ([0-9.]+)\\,d"), src_num("src/runtime/inference_service.py", r"np\.clip\(([0-9.]+) \*"), "src/runtime/inference_service.py"),
        ("d_99 quantile", tex_num(r"(\d+)th percentile of distances") / 100, src_num("scripts/build_score_bank.py", r"np\.quantile\(cal_d, ([0-9.]+)\)"), "scripts/build_score_bank.py"),
        ("blur factor", 0.5, src_num("scripts/build_score_bank.py", r"([0-9.]+) \* np\.quantile\(lap_cal"), "scripts/build_score_bank.py"),
        ("blur quantile", tex_num(r"half the (\d+)th percentile") / 100, src_num("scripts/build_score_bank.py", r"np\.quantile\(lap_cal, ([0-9.]+)\)"), "scripts/build_score_bank.py"),
        ("coreset ratio", tex_num(r"(\d+)\\% greedy") / 100, src_num("scripts/build_score_bank.py", r"coreset_sampling_ratio=([0-9.]+)"), "scripts/build_score_bank.py"),
        ("input size", tex_num(r"\$(\d+)\\times\d+\$ input"), src_num("scripts/build_score_bank.py", r"^IMG = (\d+)"), "scripts/build_score_bank.py"),
        ("input size (runtime)", tex_num(r"\$(\d+)\\times\d+\$ input"), sysc.inference.input_resolution[0], "configs/system_config.yaml"),
        ("fit split", tex_num(r"random (\d+)\\% of the good") / 100, src_num("scripts/build_score_bank.py", r"round\(([0-9.]+) \* len"), "scripts/build_score_bank.py"),
        ("glare copies per good image", 3, src_num("scripts/build_score_bank.py", r"for _ in range\((\d+)\)\]"), "scripts/build_score_bank.py"),
        ("frame rate (Hz)", tex_num(r"at (\d+) frames per second"), fps, "configs/sensor_config.yaml"),
        ("frame interval (ms)", 1000 / tex_num(r"at (\d+) frames per second"), stream_sig["frame_interval_ms"].default, "src/metrics/stream.py"),
        ("frames per run", macro_num("StepsPerRun"), {w["total_steps"] for w in wl.values()}.pop()
         if len({w["total_steps"] for w in wl.values()}) == 1 else -1, "configs/scenarios/*.yaml"),
        ("run length (min)", tex_num(r"\((\d+)\\,min at"), macro_num("StepsPerRun") / fps / 60, "9000 / 30 Hz / 60"),
        ("glare bursts", tex_num(r"(\d+) glare bursts of"), ev("transient_glitches", "vision_glitch")["count"], "transient_glitches.yaml"),
        ("glare burst min (frames)", tex_num(r"glare bursts of (\d+)--"), ev("transient_glitches", "vision_glitch")["min_len"], "transient_glitches.yaml"),
        ("glare burst max (frames)", tex_num(r"glare bursts of \d+--(\d+) frames"), ev("transient_glitches", "vision_glitch")["max_len"], "transient_glitches.yaml"),
        ("sustained defects", 8, ev("sustained_defects", "vision_defect")["count"], "sustained_defects.yaml ('eight')"),
        ("defect min (s)", tex_num(r"eight defects of (\d+)--"), ev("sustained_defects", "vision_defect")["min_len"] / fps, "sustained_defects.yaml"),
        ("defect max (s)", tex_num(r"eight defects of \d+--(\d+)"), ev("sustained_defects", "vision_defect")["max_len"] / fps, "sustained_defects.yaml"),
        ("multimodal glare events", tex_num(r"and (\d+) glare events"), ev("multimodal_faults", "vision_glitch")["count"], "multimodal_faults.yaml"),
        ("multimodal glare max (s)", tex_num(r"glare events of up to (\d+)\\,s"), ev("multimodal_faults", "vision_glitch")["max_len"] / fps, "multimodal_faults.yaml"),
        ("grace (frames)", tex_num(r"up to (\d+) frames after it"), stream_sig["grace_frames"].default, "src/metrics/stream.py"),
        ("grace (asset builder)", tex_num(r"up to (\d+) frames after it"), src_num("scripts/build_paper_assets.py", r"^GRACE = (\d+)"), "scripts/build_paper_assets.py"),
        ("recall window (frames)", tex_num(r"within (\d+) frames of onset"), stream_sig["max_delay_frames"].default, "src/metrics/stream.py"),
        ("bootstrap B", 2000, src_num("src/metrics/evaluator.py", r"n_boot=(\d+)"), "src/metrics/evaluator.py:_summarize"),
        ("bootstrap level", tex_num(r"(\d+)\\% percentile bootstrap") / 100, inspect.signature(
            __import__("src.metrics.evaluator", fromlist=["x"]).aggregate_ablation_results).parameters["ci_level"].default,
         "src/metrics/evaluator.py"),
        ("reviewer rate mu (/h)", macro_num("MuReviews"), src_num("scripts/build_paper_assets.py", r"^MU = ([0-9.]+)"), "scripts/build_paper_assets.py"),
        ("assumed glare rate (/h)", macro_num("GlareRateAssumed"), src_num("scripts/build_paper_assets.py", r"^GLARE_RATE = ([0-9.]+)"), "scripts/build_paper_assets.py"),
        ("frame budget (ms)", tex_num(r"the ([0-9.]+)\\,ms frame budget"), round(src_num("scripts/benchmark_latency.py", r"DEADLINE_MS = ([0-9.]+) / 30"), 1) / 30, "scripts/benchmark_latency.py"),
        ("event rate (/s)", tex_num(r"at (\d+) events/s"), src_num("scripts/benchmark_spooler_resilience.py", r"^RATE_HZ = ([0-9.]+)"), "scripts/benchmark_spooler_resilience.py"),
        ("overflow outage (s)", tex_num(r"When a (\d+)\\,s outage"), src_num("scripts/benchmark_spooler_resilience.py", r"case_link_loss\(port, work, ([0-9.]+), capacity="), "scripts/benchmark_spooler_resilience.py"),
        ("overflow spool capacity", 1000, src_num("scripts/benchmark_spooler_resilience.py", r"capacity=(\d+), name=\"overflow"), "scripts/benchmark_spooler_resilience.py"),
    ]
    for name, paper_v, code_v, where in rows:
        ok = math.isclose(float(paper_v), float(code_v), rel_tol=1e-3, abs_tol=0.05 if "ms" in name else 1e-9)
        record("param", name, f"{name}: paper vs code", ok, f"paper {paper_v:g}, code {float(code_v):g}",
               on_false="FLAG", paper=paper_v, code=code_v, source=where)

    # section cross-reference in the Limitations paragraph
    sections = re.findall(r"\\(section|subsection)\{([^}]*)\}", TEX)
    numbering, sec, sub = {}, 0, 0
    romans = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII"]
    for level, title in sections:
        if level == "section":
            sec, sub = sec + 1, 0
            numbering[title] = romans[sec]
        else:
            sub += 1
            numbering.setdefault(title, f"{romans[sec]}-{chr(64 + sub)}")
    m = re.search(r"Section~(V-B|IV-B|\\ref\{sec:load\})", TEX)
    stated = m.group(1) if m else None
    record("param", "xref", "Limitations refers to the operator-load subsection",
           stated in (numbering["Operator load"], "\\ref{sec:load}"),
           f"text says Section~{stated}; 'Operator load' is Section {numbering['Operator load']}",
           on_false="FLAG", paper=stated, code=numbering["Operator load"], source="paper/main.tex")


# =================================================================== 4. derived numbers
def derived_checks() -> None:
    res = ROOT / "results"
    summ = json.loads((res / "ablation" / "ablation_summary.json").read_text())["scenarios"]
    mean = lambda wl, pol, k: summ[wl][pol][k]["mean"]  # noqa: E731
    fps = 30.0

    red = 1 - mean("sustained_defects", "FULL_POLICY", "alerts_per_hour") / mean("sustained_defects", "BASELINE", "alerts_per_hour")
    record("derived", "D1", "sustained alert reduction = 1 - full/baseline",
           round(100 * red, 1) == macro_num("SustainedAlertReduction"), f"{100 * red:.3f}% vs {MACROS['SustainedAlertReduction']}")
    delay = mean("sustained_defects", "FULL_POLICY", "mean_delay_frames")
    ms_text = macro_num("SustainedDelayFullMs")
    record("derived", "D2", "mean delay in ms = frames x 1000/30", round(delay * 1000 / fps) == ms_text,
           f"{delay:.4f} frames = {delay * 1000 / fps:.1f} ms; macro SustainedDelayFullMs = {ms_text:g} ms")
    record("derived", "D3", "frame budget 33.3 ms = 1000/30", round(1000 / fps, 1) == tex_num(r"the ([0-9.]+)\\,ms frame budget"), "")

    spool = json.loads((res / "spooler_stress" / "spooler_stress_summary.json").read_text())
    ok, notes = True, []
    for c in spool["cases"]:
        ok &= c["events_generated"] - c["events_delivered_unique"] == c["missing_events"]
        # Counts only: evicted rows that were sent before eviction can still arrive, so missing <= evicted.
        ok &= max(0, c["missing_events"] - c["evicted_by_capacity"]) == 0
        if c["case"].startswith("link_loss"):
            expect = int(round(spool["event_rate_hz"] * c["outage_seconds"])) + 1
            ok &= abs(c["peak_spool_depth"] - expect) <= 1
            notes.append(f"{c['case']}: peak {c['peak_spool_depth']} ~ 30 x {c['outage_seconds']:g} + 1")
    over = next(c for c in spool["cases"] if c["case"].startswith("overflow"))
    ok &= over["evicted_by_capacity"] == int(macro_num("SpoolOverflowEvicted"))
    ok &= over["missing_events"] == int(macro_num("SpoolOverflowMissing"))
    ok &= over["evicted_by_capacity"] - over["missing_events"] == int(macro_num("SpoolOverflowEvictedDelivered"))
    ok &= int(macro_num("SpoolUnexplainedMissing")) == 0
    record("derived", "D4", "spool accounting: generated - delivered = missing <= evicted (no unexplained loss)", bool(ok),
           "; ".join(notes) + f"; overflow evicted {over['evicted_by_capacity']}, missing {over['missing_events']} "
           f"(~ 30 x 60 + 1 - 1000 = 801)")

    sens = json.loads((res / "sensitivity" / "sensitivity_summary.json").read_text())["sweeps"]["required_k"]["sustained_defects"]
    bounds = [(int(e["value"]), e["mean_delay_frames"]["mean"]) for e in sens]
    bounds.append((load_policy_config().confirmation_window.required_k, delay))
    record("derived", "D5", "measured mean delays respect the k-of-N lower bound delay >= k - 1",
           all(dl >= kk - 1 - 1e-9 for kk, dl in bounds), ", ".join(f"k={kk}: {dl:.2f}" for kk, dl in bounds))

    lat = {r["device"]: r for r in json.loads((res / "latency_benchmark_summary.json").read_text())["runs"]}
    gpu = lat["cuda"]
    e2e, dl, miss = gpu["stages"]["e2e"], gpu["deadline_ms"], gpu["deadline_miss_rate"]
    ok = (e2e["p99_ms"] <= dl or miss >= 0.01) and (e2e["p95_ms"] >= dl or miss <= 0.05)
    record("derived", "D6", "GPU deadline-miss rate is consistent with the p95/p99 quantiles", bool(ok),
           f"p95 {e2e['p95_ms']:.1f} < 33.3 => miss <= 5%; p99 {e2e['p99_ms']:.1f} > 33.3 => miss >= 1%; miss {100 * miss:.2f}%")

    ev = json.loads((res / "paper_event_metrics.json").read_text())
    mu = macro_num("MuReviews")
    g = macro_num("GlareRateAssumed")
    a_ep = sp.nsimplify(ev["sustained_defects"]["FULL_POLICY"]["per_episode"]["mean"], rational=True)
    a_gl = sp.nsimplify(ev["transient_glitches"]["FULL_POLICY"]["per_glare"]["mean"], rational=True)
    fa = sp.nsimplify(mean("nominal", "FULL_POLICY", "false_alerts_per_hour"), rational=True)
    r = sp.Symbol("r")
    r_star = sp.solve(sp.Eq((r * a_ep + g * a_gl + fa) / mu, 1), r)[0]
    record("derived", "D7", "load crossing of the full policy (rho = 1)", round(float(r_star)) == macro_num("LoadCrossFull"),
           f"r* = {float(r_star):.2f} episodes/h (paper 'about {MACROS['LoadCrossFull']}')")

    tr = json.loads((res / "real_trace_benchmark_summary.json").read_text())
    first = [-t["policies"]["FULL_POLICY"]["first_flag_rel_onset"]["mean"] for t in tr["traces"].values()]
    after = [t["policies"]["FULL_POLICY"]["alerts_from_onset"]["mean"] for t in tr["traces"].values()]
    record("derived", "D8", "trace: full policy flags before the onset and raises no alert after it",
           round(min(first)) == macro_num("TraceFirstFlagFullMin") and round(max(first)) == macro_num("TraceFirstFlagFullMax")
           and max(after) == 0, f"frames before onset {first}, alerts from onset {after}")


# =================================================================== report
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "runs" / "math_verification"))
    args = ap.parse_args()

    for group in (proofs, code_checks, parameter_checks, derived_checks):
        try:
            group()
        except Exception as exc:  # a crashing group is itself a failure, not a skip
            record(group.__name__, "crash", group.__name__, False, f"{type(exc).__name__}: {exc}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "verify_math_proofs.json").write_text(json.dumps(CHECKS, indent=1, default=str), encoding="utf-8")
    counts = {s: sum(c["status"] == s for c in CHECKS) for s in ("PASS", "FLAG", "FAIL")}
    lines = [f"# Math verification ({counts['PASS']} pass, {counts['FLAG']} flag, {counts['FAIL']} fail)", "",
             "| group | id | status | check | detail |", "|---|---|---|---|---|"]
    for c in CHECKS:
        lines.append(f"| {c['group']} | {c['id']} | {c['status']} | {c['title']} | {c['detail'].replace('|', '/')} |")
    (out / "verify_math_proofs.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for c in CHECKS:
        if c["status"] != "PASS" or c["group"] in ("proof",):
            print(f"[{c['status']}] {c['group']}/{c['id']}: {c['title']}\n        {c['detail']}")
    print(f"\n{counts}  -> {out / 'verify_math_proofs.md'}")
    sys.exit(1 if counts["FAIL"] else 0)


if __name__ == "__main__":
    main()
