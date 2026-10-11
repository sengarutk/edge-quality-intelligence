#!/usr/bin/env python3
"""Build every number, table and figure used by paper/main.tex from result files.

Inputs (all required; a missing file or key is an error, never a default):
  results/ablation/<workload>/<POLICY>_seed<unit>.json        (scripts/run_ablation_study.py)
  results/ablation_rho0.9/<workload>/<POLICY>_seed<unit>.json (scripts/run_ablation_study.py --rho 0.9)
  results/ablation/ablation_summary.json
  results/sensitivity/sensitivity_summary.json                (scripts/run_sensitivity_analysis.py)
  results/short_defect_recall.json                            (scripts/run_short_defect_recall.py)
  results/latency_benchmark_summary.json                      (scripts/benchmark_latency.py)
  results/spooler_stress/spooler_stress_summary.json          (scripts/benchmark_spooler_resilience.py)
  results/spooler_stress/delivery_failures_summary.json       (scripts/benchmark_delivery_failures.py)
  results/real_trace_benchmark_summary.json                   (scripts/run_real_trace_benchmark.py)
  results/score_bank/summary.json, optical_check_on_test.json

Statistics. An experimental unit is (category, replicate); unit id = 100 * replicate + category
index. Replicates of one category share the score pools, so they are not independent. Every
confidence interval below is a percentile bootstrap over *categories* (clusters): 7 categories are
drawn with replacement and all units of each drawn category are kept (B = 2,000).

Outputs:
  paper/generated_metrics.tex      LaTeX macros (\\newcommand)
  paper/tables/*.tex               tables included by main.tex
  paper/figures/*.pdf              figures included by main.tex
  results/paper_event_metrics.json event-normalized metrics per unit, computed here
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.experiments.workloads import build_timeline, load_workload  # noqa: E402

RES = ROOT / "results"
PAPER = ROOT / "paper"
VARIANTS = ["BASELINE", "EMA_ONLY", "EMA_KOFN", "NO_COOLDOWN", "NO_FUSION", "NO_DIVERGENCE", "NO_STATE_GATING",
            "FULL_POLICY"]
EXTERNAL = ["DELAY_TIMER", "EMA_HYSTERESIS", "DECISION_FUSION"]
POLICIES = VARIANTS + EXTERNAL
RHO_DIR = "ablation_rho0.9"
VISA_DIR = "ablation_visa"
VISA_RHO_DIR = "ablation_visa_rho0.9"
GRACE = 15
MU = 60.0  # reviews per hour, one operator (about one minute per review); an assumption
GLARE_RATE = 30.0  # glare bursts per hour; an assumption
B_BOOT = 2000
CATEGORIES = ["bottle", "cable", "carpet", "grid", "hazelnut", "leather", "metal_nut"]
STYLE = {"BASELINE": ("#2a78d6", "o"), "EMA_KOFN": ("#eb6834", "s"), "DELAY_TIMER": ("#7d3ac1", "v"),
         "DECISION_FUSION": ("#1baf7a", "^"), "FULL_POLICY": ("#eda100", "D")}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

# Table columns: (label, workload, metric, digits). Metrics prefixed "ev:" are event-normalized.
COLUMNS = [
    ("nominal FA/h", "nominal", "false_alerts_per_hour", 1),
    ("glare alerts/burst", "transient_glitches", "ev:per_glare", 2),
    ("alerts/ep.", "sustained_defects", "ev:per_episode", 2),
    ("delay (fr.)", "sustained_defects", "mean_delay_frames", 1),
    ("defect recall", "sustained_defects", "routing_recall", 2),
    ("fault recall", "multimodal_faults", "routing_recall", 2),
    ("crit. FA/h", "multimodal_faults", "false_high_alerts_per_hour", 1),
    ("degraded FA/h", "degraded_inputs", "false_alerts_per_hour", 1),
    ("states FA/h", "state_transitions", "false_alerts_per_hour", 1),
]
HIGHER_IS_BETTER = {"routing_recall"}


def need(d: Dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            raise KeyError(f"missing key {'/'.join(keys)}")
        d = d[k]
    return d


def load(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{path} is required; run the corresponding experiment first")
    return json.loads(path.read_text(encoding="utf-8"))


def spans(mask: np.ndarray):
    idx = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return list(zip(idx[::2], idx[1::2]))


# ------------------------------------------------------------------ per-unit data
def load_units(dirname: str) -> Dict[Tuple[str, str], Dict[int, Dict[str, Any]]]:
    """{(workload, policy): {unit: metrics + event metrics}} from the per-run files of one directory."""
    files = sorted((RES / dirname).glob("*/*_seed*.json"))
    if not files:
        raise FileNotFoundError(f"no runs in results/{dirname}")
    wl_cache, tl_cache = {}, {}
    out: Dict[Tuple[str, str], Dict[int, Dict[str, Any]]] = {}
    for f in files:
        r = json.loads(f.read_text(encoding="utf-8"))
        wl, unit = r["workload"], int(r["seed"])
        if wl not in wl_cache:
            wl_cache[wl] = load_workload(ROOT / "configs" / "scenarios" / f"{wl}.yaml")
        if (wl, unit) not in tl_cache:
            tl_cache[(wl, unit)] = build_timeline(wl_cache[wl], unit)
        tl = tl_cache[(wl, unit)]
        covered = np.zeros(tl.n, bool)
        for a, b in spans(tl.actionable):
            covered[a:min(tl.n, b + GRACE)] = True
        alerts = np.zeros(tl.n, bool)
        alerts[r["alert_steps"]] = True
        m = dict(r["metrics"])
        m["category"] = r["category"]
        m["alert_reasons"] = r["alert_reasons"]
        bursts = [(a, b) for a, b in spans(tl.vision_source == 2) if tl.machine_state[a] == "RUNNING"]
        if bursts:
            n = sum(int(alerts[a:min(tl.n, b + GRACE)][~covered[a:min(tl.n, b + GRACE)]].sum()) for a, b in bursts)
            m["ev:per_glare"] = n / len(bursts)
        eps = spans(tl.actionable)
        if eps:
            m["ev:per_episode"] = float(np.mean([alerts[a:min(tl.n, b + GRACE)].sum() for a, b in eps]))
        out.setdefault((wl, r["policy_mode"]), {})[unit] = m
    return out


def unit_values(units, wl: str, pol: str, metric: str) -> Dict[int, float]:
    cell = need({f"{k[0]}|{k[1]}": v for k, v in units.items()}, f"{wl}|{pol}")
    return {u: float(m[metric]) for u, m in cell.items() if m.get(metric) is not None}


def cluster_boot(values: Dict[int, float], stat: Callable[[np.ndarray], float] = np.mean,
                 seed: int = 2026) -> Dict[str, float]:
    """Mean over units with a percentile bootstrap that resamples categories (unit id % 100)."""
    if not values:
        raise ValueError("no values")
    by_cat: Dict[int, List[float]] = {}
    for u, v in values.items():
        by_cat.setdefault(u % 100, []).append(v)
    cats = sorted(by_cat)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(B_BOOT):
        pick = rng.choice(len(cats), size=len(cats), replace=True)
        boots.append(stat(np.concatenate([by_cat[cats[i]] for i in pick])))
    arr = np.array(list(values.values()))
    return {"mean": float(stat(arr)), "lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5)),
            "n_units": len(values), "n_categories": len(cats)}


def mean_of(units, wl, pol, metric) -> float:
    return float(np.mean(list(unit_values(units, wl, pol, metric).values())))


# ------------------------------------------------------------------ formatting
def fmt(x: Optional[float], nd: int = 1) -> str:
    if x is None:
        return "--"
    if abs(x) >= 1000:
        return f"{x:,.0f}".replace(",", "{,}")
    return f"{x:.{nd}f}"


def pct(x: float, nd: int = 1) -> str:
    return f"{100 * x:.{nd}f}\\%"


def tt(name: str) -> str:
    return "\\texttt{" + name.replace("_", "\\_") + "}"


# ------------------------------------------------------------------ tables
def table_ablation(units, policies: List[str], external_from: Optional[str] = "DELAY_TIMER") -> str:
    rows = []
    for pol in policies:
        if pol == external_from:
            rows.append("\\midrule")
        cells = []
        for _, wl, metric, nd in COLUMNS:
            vals = unit_values(units, wl, pol, metric)
            cells.append(fmt(float(np.mean(list(vals.values()))), nd) if vals else "--")
        rows.append(f"{tt(pol)} & " + " & ".join(cells) + " \\\\")
    return "\n".join([
        "\\begin{tabular}{@{}lrrrrrrrrr@{}}", "\\toprule",
        " & Nominal & Glare & \\multicolumn{3}{c}{Sustained defects} & \\multicolumn{2}{c}{Multi-modal faults} & Degraded & States \\\\",
        "\\cmidrule(lr){4-6}\\cmidrule(lr){7-8}",
        "Policy & FA/h & alerts/burst & alerts/ep. & delay (fr.) & recall & recall & crit.\\ FA/h & FA/h & FA/h \\\\",
        "\\midrule", *rows, "\\bottomrule", "\\end{tabular}",
    ])


def paired(units, wl, metric, ref: str, other: str) -> Dict[str, Any]:
    """other - ref per unit (common units), with a category-cluster bootstrap CI and category wins of ref."""
    a, b = unit_values(units, wl, ref, metric), unit_values(units, wl, other, metric)
    common = sorted(set(a) & set(b))
    d = {u: b[u] - a[u] for u in common}
    res = cluster_boot(d)
    cat_mean: Dict[int, List[float]] = {}
    for u, v in d.items():
        cat_mean.setdefault(u % 100, []).append(v)
    # ref (FULL_POLICY) is better in a category when its category mean is lower (higher for recall).
    better = sum(1 for v in cat_mean.values() if (np.mean(v) < 0 if metric in HIGHER_IS_BETTER else np.mean(v) > 0))
    worse = sum(1 for v in cat_mean.values() if (np.mean(v) > 0 if metric in HIGHER_IS_BETTER else np.mean(v) < 0))
    res["perm_p"] = sign_flip_p([float(np.mean(v)) for v in cat_mean.values()])
    res["category_deltas"] = {CATS_BY_DIR.get(c, str(c)): float(np.mean(v)) for c, v in sorted(cat_mean.items())}
    ties = len(cat_mean) - better - worse
    res.update({"ref_better_categories": better, "ref_worse_categories": worse, "tie_categories": ties,
                "categories": len(cat_mean)})
    return res


CATS_BY_DIR: Dict[int, str] = {}


def sign_flip_p(deltas: List[float]) -> float:
    """Exact two-sided sign-flip permutation p-value of the mean of category-level paired differences.

    With n categories there are 2^n sign assignments; the smallest attainable p is 2 / 2^n.
    """
    d = np.asarray(deltas, dtype=float)
    if np.allclose(d, 0):
        return 1.0
    n = len(d)
    signs = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1
    stats = np.abs((signs * d).mean(axis=1))
    return float(np.mean(stats >= abs(d.mean()) - 1e-12))


def fmt_p(p: float) -> str:
    """Complete math-mode expression, e.g. $p=0.016$ or $p<0.001$."""
    return f"$p={p:.3f}$" if p >= 0.001 else "$p<0.001$"


def fmt_ci(r: Dict[str, float], nd: int) -> str:
    return f"{fmt(r['mean'], nd)} [{fmt(r['lo'], nd)}, {fmt(r['hi'], nd)}]"


def table_effects(units, comparators: List[str]) -> Tuple[str, Dict]:
    """Rows: metrics. Columns: comparator - FULL_POLICY, mean difference [95% category-cluster CI] (wins)."""
    data = {}
    rows = []
    for label, wl, metric, nd in COLUMNS:
        cells = []
        for c in comparators:
            r = paired(units, wl, metric, "FULL_POLICY", c)
            data[(metric, wl, c)] = r
            cells.append(f"{fmt_ci(r, nd)} ({r['ref_better_categories']}:{r['ref_worse_categories']}; {fmt_p(r['perm_p'])})")
        rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
    head = " & ".join(tt(c) for c in comparators)
    return "\n".join([f"\\begin{{tabular}}{{@{{}}l{'r' * len(comparators)}@{{}}}}", "\\toprule",
                      f"Metric & {head} \\\\", "\\midrule", *rows, "\\bottomrule", "\\end{tabular}"]), data


def table_per_category(units) -> str:
    rows = []
    for ci, cat in enumerate(CATEGORIES):
        cells = []
        for _, wl, metric, nd in COLUMNS:
            v = [x for u, x in unit_values(units, wl, "FULL_POLICY", metric).items() if u % 100 == ci]
            cells.append(fmt(float(np.mean(v)), nd) if v else "--")
        rows.append(f"{cat.replace('_', chr(92) + '_')} & " + " & ".join(cells) + " \\\\")
    cis = []
    for _, wl, metric, nd in COLUMNS:
        r = cluster_boot(unit_values(units, wl, "FULL_POLICY", metric))
        cis.append(f"[{fmt(r['lo'], nd)}, {fmt(r['hi'], nd)}]")
    return "\n".join([
        "\\begin{tabular}{@{}lrrrrrrrrr@{}}", "\\toprule",
        " & Nominal & Glare & \\multicolumn{3}{c}{Sustained defects} & \\multicolumn{2}{c}{Multi-modal faults} & Degraded & States \\\\",
        "\\cmidrule(lr){4-6}\\cmidrule(lr){7-8}",
        "Category & FA/h & alerts/burst & alerts/ep. & delay (fr.) & recall & recall & crit.\\ FA/h & FA/h & FA/h \\\\",
        "\\midrule", *rows, "\\midrule",
        "95\\% CI & " + " & ".join(cis) + " \\\\",
        "\\bottomrule", "\\end{tabular}",
    ])


def table_durability(spool) -> str:
    names = {"link_loss_30s": "Link loss 30\\,s", "link_loss_60s": "Link loss 60\\,s",
             "link_loss_120s": "Link loss 120\\,s", "broker_restart_30s": "Broker restart 30\\,s",
             "publisher_crash_sigkill": "Publisher SIGKILL", "overflow_60s_capacity_1000": "Overflow (cap.\\ 1{,}000)"}
    rows = []
    for c in need(spool, "cases"):
        rows.append(f"{names[c['case']]} & {fmt(c['events_generated'], 0)} & {c['missing_events']} & "
                    f"{c['evicted_by_capacity']} & {c['duplicate_deliveries']} & {c['order_violations']} & "
                    f"{fmt(c['peak_spool_depth'], 0)} & {c['drain_seconds']:.2f} \\\\")
    return "\n".join(["\\begin{tabular}{@{}lrrrrrrr@{}}", "\\toprule",
                      "Fault & Events & Miss. & Evict. & Dup. & Order & Peak & Drain (s) \\\\", "\\midrule",
                      *rows, "\\bottomrule", "\\end{tabular}"])


def table_delivery_failures(df) -> str:
    sub_names = {"clean_session_10s": "Clean session",
                 "persistent_10s_default_queue": "Persistent, queue 1{,}000",
                 "persistent_60s_default_queue": "Persistent, queue 1{,}000",
                 "persistent_60s_unlimited_queue": "Persistent, unlimited"}
    rows = [f"{sub_names[c['case']]} & {c['subscriber_down_seconds']:.0f} & {fmt(c['events_generated_while_down'], 0)} & "
            f"{fmt(c['missing_events'], 0)} \\\\" for c in need(df, "subscriber_crash")]
    crash = {c["mode"]: c for c in need(df, "inspection_crash")}
    rows2 = [f"{label} & {crash[k]['kills']} & {fmt(crash[k]['decisions_handed_off'], 0)} & "
             f"{crash[k]['decisions_lost']} ({crash[k]['kills_with_loss']}) \\\\"
             for k, label in (("synchronous", "Synchronous"), ("writer_thread", "Writer thread"))]
    return "\n".join(["\\begin{tabular}{@{}lrrr@{}}", "\\toprule",
                      "Subscriber SIGKILL & Down (s) & Sent while down & Missing \\\\", "\\midrule", *rows,
                      "\\midrule",
                      "Inspection SIGKILL & Kills & Handed off & Lost (kills) \\\\", "\\midrule", *rows2,
                      "\\bottomrule", "\\end{tabular}"])


def latency_runs(lat) -> Dict[Tuple[str, str], Dict]:
    return {(r["device"], r["persistence"]): r for r in need(lat, "runs")}


def table_latency(lat) -> str:
    runs_ = latency_runs(lat)
    gs, ga, ca = runs_[("cuda", "sync")], runs_[("cuda", "async")], runs_[("cpu", "async")]
    labels = {"decode": "PNG decode", "vision": "Optical check + PatchCore", "sensor": "Sensor step",
              "policy": "Policy", "spool": "Spool insert", "audit": "Audit insert",
              "persist": "Enqueue to writer", "e2e": "End to end"}

    def cell(run, k, key, nd):
        st = run["stages"].get(k)
        return "--" if st is None else f"{st[key]:.{nd}f}"

    rows = []
    for k, lab in labels.items():
        sep = "\\midrule\n" if k == "e2e" else ""
        rows.append(f"{sep}{lab} & {cell(gs, k, 'mean_ms', 2)} & {cell(gs, k, 'p99_ms', 2)} & {cell(gs, k, 'max_ms', 1)} & "
                    f"{cell(ga, k, 'mean_ms', 2)} & {cell(ga, k, 'p99_ms', 2)} & {cell(ga, k, 'max_ms', 1)} & "
                    f"{cell(ca, k, 'mean_ms', 2)} & {cell(ca, k, 'p99_ms', 2)} \\\\")
    miss = (f"Deadline misses & \\multicolumn{{3}}{{r}}{{{pct(gs['deadline_miss_rate'], 2)}}} & "
            f"\\multicolumn{{3}}{{r}}{{{pct(ga['deadline_miss_rate'], 2)}}} & \\multicolumn{{2}}{{r}}{{{pct(ca['deadline_miss_rate'], 0)}}} \\\\")
    return "\n".join(["\\begin{tabular}{@{}lrrrrrrrr@{}}", "\\toprule",
                      " & \\multicolumn{3}{c}{GPU, synchronous} & \\multicolumn{3}{c}{GPU, writer thread} & \\multicolumn{2}{c}{CPU, writer thread} \\\\",
                      "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-9}",
                      "Stage (ms) & mean & p99 & max & mean & p99 & max & mean & p99 \\\\", "\\midrule", *rows,
                      miss, "\\bottomrule", "\\end{tabular}"])


def table_trace(tr) -> str:
    rows = []
    names = {"bearing_proxy": "Bearing proxy", "thermal_creep_proxy": "Thermal-creep proxy"}
    for t, info in need(tr, "traces").items():
        for pol in ("BASELINE", "DELAY_TIMER", "DECISION_FUSION", "NO_FUSION", "FULL_POLICY"):
            s = need(info, "policies", pol)
            first = s.get("first_flag_rel_onset", {}).get("mean")
            rows.append(f"{names[t] if pol == 'BASELINE' else ''} & {tt(pol)} & "
                        f"{fmt(first, 0) if first is not None else 'never'} & "
                        f"{fmt(need(s, 'alerts_before_onset', 'mean'), 1)} & {fmt(need(s, 'alerts_from_onset', 'mean'), 1)} & "
                        f"{fmt(need(s, 'high_alerts_from_onset', 'mean'), 1)} \\\\")
        rows.append("\\midrule")
    rows[-1] = "\\bottomrule"
    return "\n".join(["\\begin{tabular}{@{}llrrrr@{}}", "\\toprule",
                      " & & first flag & \\multicolumn{2}{c}{alerts} & HIGH alerts \\\\",
                      "\\cmidrule(lr){4-5}",
                      "Trace & Policy & (fr.\\ vs.\\ onset) & before & from onset & from onset \\\\", "\\midrule", *rows,
                      "\\end{tabular}"])


# ---------------------------------------------------------------- figures
def style_axes(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=7)


def load_inputs(units, pol) -> Tuple[float, float, float]:
    a_ep = mean_of(units, "sustained_defects", pol, "ev:per_episode")
    a_gl = mean_of(units, "transient_glitches", pol, "ev:per_glare")
    fa = mean_of(units, "nominal", pol, "false_alerts_per_hour")
    return a_ep, a_gl, fa


def fig_operator_load(units, path: Path) -> Dict[str, float]:
    """rho(r) = [r * alerts/episode + g * alerts/glare burst + nominal FA/h] / mu."""
    r = np.linspace(0, 30, 121)
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    crossings = {}
    for i, (pol, (color, marker)) in enumerate(STYLE.items()):
        a_ep, a_gl, fa = load_inputs(units, pol)
        rho = (r * a_ep + GLARE_RATE * a_gl + fa) / MU
        ax.plot(r, rho, color=color, linewidth=1.8, marker=marker, markevery=(4 * i, 20), markersize=4,
                label=pol.replace("_", " ").lower())
        crossings[pol] = max(0.0, (MU - GLARE_RATE * a_gl - fa) / a_ep)
    ax.axhline(1.0, color=INK, linewidth=0.8, linestyle="--")
    ax.text(29.5, 1.1, "$\\rho = 1$", ha="right", va="bottom", fontsize=7, color=INK)
    ax.set_yscale("log")
    ax.set_xlabel("Defect episodes per hour", fontsize=8, color=INK)
    ax.set_ylabel("Reviewer utilization $\\rho$", fontsize=8, color=INK)
    style_axes(ax)
    ax.legend(fontsize=6.2, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.27))
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return crossings


def fig_category_deltas(effects, path: Path) -> None:
    """Raw category-level paired differences (comparator - FULL_POLICY) for four metrics."""
    panels = [("ev:per_glare", "transient_glitches", "Glare alerts/burst"),
              ("false_high_alerts_per_hour", "multimodal_faults", "Crit. FA/h (multi-modal)"),
              ("false_alerts_per_hour", "degraded_inputs", "FA/h (degraded)"),
              ("false_alerts_per_hour", "state_transitions", "FA/h (states)")]
    comps = EXTERNAL
    fig, axes = plt.subplots(1, len(panels), figsize=(7.0, 1.9))
    for ax, (metric, wl, title) in zip(axes, panels):
        for i, c in enumerate(comps):
            vals = list(effects[(metric, wl, c)]["category_deltas"].values())
            x = i + np.linspace(-0.18, 0.18, len(vals))
            ax.scatter(x, vals, s=10, color=STYLE.get(c, ("#555555", "o"))[0], zorder=3)
            ax.plot([i - 0.25, i + 0.25], [np.mean(vals)] * 2, color=INK, linewidth=1.2)
        ax.axhline(0, color=INK2, linewidth=0.8, linestyle="--")
        ax.set_xticks(range(len(comps)))
        ax.set_xticklabels(["timer", "hyst.", "fusion"], fontsize=7)
        ax.set_title(title, fontsize=7, color=INK)
        style_axes(ax)
    axes[0].set_ylabel("baseline $-$ full policy", fontsize=7, color=INK)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig_sensitivity(sens, path: Path) -> None:
    params = [("required_k", "$k$"), ("cross_modal_divergence", "$\\tau_{div}$"),
              ("cooldown_steps", "$T_{cool}$"), ("alpha_vision", "$\\alpha_v$")]
    rows = [("multimodal_faults", "false_high_alerts_per_hour", "Crit. FA/h\n(multi-modal)"),
            ("transient_glitches", "false_alerts_per_hour", "FA/h\n(glare)"),
            ("sustained_defects", "mean_delay_frames", "Delay (fr.)\n(sustained)")]
    defaults = {"required_k": 4, "cross_modal_divergence": 0.45, "cooldown_steps": 15, "alpha_vision": 0.35}
    fig, axes = plt.subplots(len(rows), len(params), figsize=(7.0, 3.6), sharey="row")
    for i, (wl, metric, ylab) in enumerate(rows):
        for j, (p, plab) in enumerate(params):
            ax = axes[i, j]
            pts = need(sens, "sweeps", p, wl)
            x = [e["value"] for e in pts]
            y = [need(e, metric, "mean") for e in pts]
            lo = [need(e, metric, "ci_lower") for e in pts]
            hi = [need(e, metric, "ci_upper") for e in pts]
            ax.fill_between(x, lo, hi, color="#2a78d6", alpha=0.18, linewidth=0)
            ax.plot(x, y, color="#2a78d6", linewidth=2, marker="o", markersize=4)
            ax.axvline(defaults[p], color=INK2, linewidth=0.8, linestyle=":")
            style_axes(ax)
            if i == len(rows) - 1:
                ax.set_xlabel(plab, fontsize=8, color=INK)
            if j == 0:
                ax.set_ylabel(ylab, fontsize=7, color=INK)
    fig.tight_layout(h_pad=0.4, w_pad=0.4)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ----------------------------------------------------------------- macros
def macros(units, units_rho, effects, summary, sens, lat, spool, tr, bank, optical, crossings, short,
           units_visa, units_visa_rho, effects_visa, visa_bank, dfail) -> Dict[str, str]:
    m: Dict[str, str] = {}
    design = need(summary, "design")
    m["NumVariants"] = str(len(VARIANTS))
    m["NumExternal"] = str(len(EXTERNAL))
    m["NumPolicies"] = str(len(design["policy_modes"]))
    m["NumWorkloads"] = str(len(design["workloads"]))
    m["NumCategories"] = str(len(design["categories"]))
    m["NumReplicates"] = str(len(design["replicates"]))
    m["UnitsPerCell"] = str(design["units_per_cell"])
    m["StepsPerRun"] = fmt(design["steps_per_run"], 0)
    m["TotalRuns"] = fmt(len(list((RES / "ablation").glob("*/*_seed*.json"))), 0)
    m["TotalRunsRho"] = fmt(len(list((RES / RHO_DIR).glob("*/*_seed*.json"))), 0)
    m["BootB"] = fmt(B_BOOT, 0)

    aurocs = [c["image_auroc"] for c in need(bank, "categories")]
    m["BankAurocMin"], m["BankAurocMax"] = f"{min(aurocs):.3f}", f"{max(aurocs):.3f}"
    med_def = [c["median_norm_score_defect"] for c in bank["categories"]]
    m["DefectMedianBelowHigh"] = str(sum(v < 0.8 for v in med_def))
    m["GlareMedianMin"] = f"{min(c['median_norm_score_glare'] for c in bank['categories']):.2f}"
    goods = [c["n_test_nominal"] for c in bank["categories"]]
    m["GoodPoolMin"], m["GoodPoolMax"] = str(min(goods)), str(max(goods))
    worst = max(optical.items(), key=lambda kv: kv[1]["flag_rate_defective"])
    m["OpticalWorstCategory"] = worst[0].replace("_", "\\_")
    m["OpticalWorstRate"] = pct(worst[1]["flag_rate_defective"])
    m["OpticalWorstTypes"] = ", ".join(t.replace("_", "\\_") for t in worst[1]["flagged_by_defect_type"])
    m["OpticalNominalFlagMax"] = pct(max(v["flag_rate_nominal"] for v in optical.values()))

    g = lambda wl, pol, k, u=units: mean_of(u, wl, pol, k)  # noqa: E731
    cb = lambda wl, pol, k, u=units: cluster_boot(unit_values(u, wl, pol, k))  # noqa: E731

    def mac_ci(name, wl, pol, k, nd, u=units):
        r = cb(wl, pol, k, u)
        m[name] = fmt(r["mean"], nd)
        m[name + "Lo"] = fmt(r["lo"], nd)
        m[name + "Hi"] = fmt(r["hi"], nd)

    mac_ci("NominalFABaseline", "nominal", "BASELINE", "false_alerts_per_hour", 0)
    m["NominalFAFull"] = fmt(g("nominal", "FULL_POLICY", "false_alerts_per_hour"), 1)
    m["NominalFAEmaOnly"] = fmt(g("nominal", "EMA_ONLY", "false_alerts_per_hour"), 1)
    per_cat_nom = {CATEGORIES[c]: np.mean([v for u, v in unit_values(units, "nominal", "BASELINE", "false_alerts_per_hour").items() if u % 100 == c]) for c in range(7)}
    m["NominalFABaselineZeroCats"] = str(sum(v == 0 for v in per_cat_nom.values()))
    m["NominalFABaselineMaxCat"] = max(per_cat_nom, key=per_cat_nom.get).replace("_", "\\_")
    m["NominalFABaselineMaxCatVal"] = fmt(max(per_cat_nom.values()), 0)
    for pol, tag in (("BASELINE", "Baseline"), ("EMA_KOFN", "EmaKofn"), ("FULL_POLICY", "Full"),
                     ("DELAY_TIMER", "Timer"), ("EMA_HYSTERESIS", "Hyst"), ("DECISION_FUSION", "Fusion")):
        mac_ci(f"GlarePerBurst{tag}", "transient_glitches", pol, "ev:per_glare", 2)
        mac_ci(f"AlertsPerEpisode{tag}", "sustained_defects", pol, "ev:per_episode", 2)
        m[f"SustainedDelay{tag}"] = fmt(g("sustained_defects", pol, "mean_delay_frames"), 1)
        m[f"MultiRecall{tag}"] = fmt(g("multimodal_faults", pol, "routing_recall"), 2)
        m[f"DefectRecall{tag}"] = fmt(g("sustained_defects", pol, "routing_recall"), 2)
        m[f"CritFA{tag}"] = fmt(g("multimodal_faults", pol, "false_high_alerts_per_hour"), 1)
        m[f"DegradedFA{tag}"] = fmt(g("degraded_inputs", pol, "false_alerts_per_hour"), 1)
        m[f"StateFA{tag}"] = fmt(g("state_transitions", pol, "false_alerts_per_hour"), 1)
        m[f"NominalFA{tag}"] = fmt(g("nominal", pol, "false_alerts_per_hour"), 1)
    m["SustainedAlertsBaseline"] = fmt(g("sustained_defects", "BASELINE", "alerts_per_hour"), 0)
    m["SustainedAlertsFull"] = fmt(g("sustained_defects", "FULL_POLICY", "alerts_per_hour"), 0)
    red = 1 - g("sustained_defects", "FULL_POLICY", "alerts_per_hour") / g("sustained_defects", "BASELINE", "alerts_per_hour")
    m["SustainedAlertReduction"] = pct(red)
    m["SustainedDelayFullMs"] = f"{g('sustained_defects', 'FULL_POLICY', 'mean_delay_frames') * 1000 / 30:.0f}"
    m["MultiRecallNoFusion"] = fmt(g("multimodal_faults", "NO_FUSION", "routing_recall"), 2)
    dl = unit_values(units, "sustained_defects", "FULL_POLICY", "mean_delay_frames")
    cat_delay = [np.mean([v for u, v in dl.items() if u % 100 == c]) for c in range(len(CATEGORIES))]
    m["PerCatDelayMin"], m["PerCatDelayMax"] = fmt(min(cat_delay), 1), fmt(max(cat_delay), 1)
    mac_ci("CritFAFullCI", "multimodal_faults", "FULL_POLICY", "false_high_alerts_per_hour", 1)
    m["CritFANoDiv"] = fmt(g("multimodal_faults", "NO_DIVERGENCE", "false_high_alerts_per_hour"), 1)
    m["StateFANoGating"] = fmt(g("state_transitions", "NO_STATE_GATING", "false_alerts_per_hour"), 1)

    reasons: Dict[str, int] = {}
    for u, rec in units[("degraded_inputs", "FULL_POLICY")].items():
        for k, v in rec["alert_reasons"].items():
            reasons[k] = reasons.get(k, 0) + v
    m["DegradedAlertsTotal"] = str(sum(reasons.values()))
    m["DegradedOpticalAlerts"] = str(reasons.get("OPTICAL_DEGRADATION_FALLBACK", 0))
    m["DegradedSensorAlerts"] = str(reasons.get("SUSTAINED_SENSOR_ANOMALY", 0))
    # Degraded-input false alerts without the camera-maintenance requests (sustained blur/darkness),
    # per unit: every OPTICAL_DEGRADATION_FALLBACK alert lies outside the defect windows.
    no_maint = {}
    for u, rec in units[("degraded_inputs", "FULL_POLICY")].items():
        opt = rec["alert_reasons"].get("OPTICAL_DEGRADATION_FALLBACK", 0)
        fa, fa_h = rec["false_alerts"], rec["false_alerts_per_hour"]
        no_maint[u] = fa_h * (fa - opt) / fa if fa else 0.0
    m["DegradedFAFullNoMaint"] = fmt(float(np.mean(list(no_maint.values()))), 1)

    # Paired effects of FULL_POLICY vs. the external baselines (comparator - FULL; category-cluster CI).
    for (metric, wl, comp), r in effects.items():
        key = {"false_alerts_per_hour": "FA", "false_high_alerts_per_hour": "CritFA", "ev:per_glare": "Glare",
               "ev:per_episode": "AlertsEp", "mean_delay_frames": "Delay", "routing_recall": "Recall"}[metric]
        wtag = {"nominal": "Nom", "transient_glitches": "", "sustained_defects": "", "multimodal_faults": "Multi",
                "degraded_inputs": "Degr", "state_transitions": "State"}[wl]
        ctag = {"DELAY_TIMER": "Timer", "EMA_HYSTERESIS": "Hyst", "DECISION_FUSION": "Fusion", "BASELINE": "Baseline"}[comp]
        base = f"Eff{key}{wtag}{ctag}"
        nd = 2 if metric in ("ev:per_glare", "ev:per_episode", "routing_recall") else 1
        m[base] = fmt(r["mean"], nd)
        m[base + "Lo"] = fmt(r["lo"], nd)
        m[base + "Hi"] = fmt(r["hi"], nd)
        m[base + "Wins"] = str(r["ref_better_categories"])
        m[base + "Losses"] = str(r["ref_worse_categories"])
        m[base + "P"] = fmt_p(r["perm_p"])

    # Temporally correlated visual scores (rho = 0.9).
    for pol, tag in (("BASELINE", "Baseline"), ("EMA_KOFN", "EmaKofn"), ("FULL_POLICY", "Full"),
                     ("DELAY_TIMER", "Timer"), ("DECISION_FUSION", "Fusion"), ("EMA_HYSTERESIS", "Hyst")):
        m[f"RhoNominalFA{tag}"] = fmt(g("nominal", pol, "false_alerts_per_hour", units_rho), 1)
        m[f"RhoGlare{tag}"] = fmt(g("transient_glitches", pol, "ev:per_glare", units_rho), 2)
        m[f"RhoAlertsEp{tag}"] = fmt(g("sustained_defects", pol, "ev:per_episode", units_rho), 2)
        m[f"RhoDelay{tag}"] = fmt(g("sustained_defects", pol, "mean_delay_frames", units_rho), 1)
        m[f"RhoCritFA{tag}"] = fmt(g("multimodal_faults", pol, "false_high_alerts_per_hour", units_rho), 1)
        m[f"RhoRecall{tag}"] = fmt(g("multimodal_faults", pol, "routing_recall", units_rho), 2)
        m[f"RhoDefectRecall{tag}"] = fmt(g("sustained_defects", pol, "routing_recall", units_rho), 2)

    # sensitivity endpoints
    kpts = need(sens, "sweeps", "required_k", "multimodal_faults")
    m["SensKLow"], m["SensKHigh"] = str(int(kpts[0]["value"])), str(int(kpts[-1]["value"]))
    m["SensCritKLow"] = fmt(kpts[0]["false_high_alerts_per_hour"]["mean"])
    m["SensCritKHigh"] = fmt(kpts[-1]["false_high_alerts_per_hour"]["mean"])
    gpts = need(sens, "sweeps", "required_k", "transient_glitches")
    m["SensGlareKLow"] = fmt(gpts[0]["false_alerts_per_hour"]["mean"], 0)
    m["SensGlareKHigh"] = fmt(gpts[-1]["false_alerts_per_hour"]["mean"], 0)
    dpts = need(sens, "sweeps", "required_k", "sustained_defects")
    m["SensDelayKLow"] = fmt(dpts[0]["mean_delay_frames"]["mean"])
    m["SensDelayKHigh"] = fmt(dpts[-1]["mean_delay_frames"]["mean"])
    crit = [e["false_high_alerts_per_hour"]["mean"] for p in sens["sweeps"].values() for e in p["multimodal_faults"]]
    m["SensCritMin"], m["SensCritMax"] = fmt(min(crit)), fmt(max(crit))
    rec = [e["routing_recall"]["mean"] for p in sens["sweeps"].values() for e in p["multimodal_faults"]]
    m["SensRecallMin"] = fmt(min(rec), 2)
    for p, tag in (("cross_modal_divergence", "Div"), ("alpha_vision", "Alpha"), ("cooldown_steps", "Cool")):
        pts = need(sens, "sweeps", p, "multimodal_faults")
        m[f"SensCrit{tag}Lo"] = fmt(pts[0]["false_high_alerts_per_hour"]["mean"])
        m[f"SensCrit{tag}Hi"] = fmt(pts[-1]["false_high_alerts_per_hour"]["mean"])
        gp = need(sens, "sweeps", p, "transient_glitches")
        m[f"SensGlare{tag}Lo"] = fmt(gp[0]["false_alerts_per_hour"]["mean"], 0)
        m[f"SensGlare{tag}Hi"] = fmt(gp[-1]["false_alerts_per_hour"]["mean"], 0)

    m["GlareRateAssumed"] = fmt(GLARE_RATE, 0)
    m["MuReviews"] = fmt(MU, 0)
    for pol, tag in (("FULL_POLICY", "Full"), ("EMA_KOFN", "EmaKofn"), ("DELAY_TIMER", "Timer"),
                     ("DECISION_FUSION", "Fusion"), ("BASELINE", "Baseline")):
        m[f"LoadCross{tag}"] = fmt(crossings[pol], 0)
    a_ep, a_gl, fa = load_inputs(units, "FULL_POLICY")
    m["PrecFullOneEp"] = pct(a_ep / (a_ep + GLARE_RATE * a_gl + fa), 0)
    a_ep, a_gl, fa = load_inputs(units, "DELAY_TIMER")
    m["PrecTimerOneEp"] = pct(a_ep / (a_ep + GLARE_RATE * a_gl + fa), 0)

    # recall against defect duration
    k, n = need(short, "k_of_n")
    m["ShortK"], m["ShortN"] = str(k), str(n)
    m["ShortRecallTarget"] = fmt(need(short, "recall_target"), 2)
    for pol, tag in (("BASELINE", "Baseline"), ("FULL_POLICY", "Full"), ("DELAY_TIMER", "Timer"),
                     ("DECISION_FUSION", "Fusion"), ("EMA_HYSTERESIS", "Hyst")):
        by_len = {e["length"]: e["recall"]["mean"] for e in need(short, "policies", pol, "per_length")}
        for name, L in (("One", 1), ("Two", 2), ("Three", 3), ("Five", 5), ("Ten", 10)):
            m[f"ShortRecall{tag}{name}"] = fmt(by_len[L], 2)
        min_len = need(short, "policies", pol, "min_length_for_target_recall")
        m[f"ShortMinLen{tag}"] = str(min_len) if min_len is not None else "--"
        if min_len is not None:
            m[f"ShortMinLen{tag}Ms"] = f"{min_len * 1000 / 30:.0f}"

    runs_ = latency_runs(lat)
    gs, ga, cs, ca = runs_[("cuda", "sync")], runs_[("cuda", "async")], runs_[("cpu", "sync")], runs_[("cpu", "async")]
    m["LatGpuName"] = gs["device_name"].replace("NVIDIA GeForce ", "")
    m["LatCycles"] = fmt(gs["cycles"], 0)
    m["LatCpuCycles"] = fmt(cs["cycles"], 0)
    for run, tag in ((gs, "GpuSync"), (ga, "GpuAsync"), (cs, "CpuSync"), (ca, "CpuAsync")):
        e = run["stages"]["e2e"]
        m[f"Lat{tag}Mean"] = f"{e['mean_ms']:.1f}"
        m[f"Lat{tag}PNinetyFive"] = f"{e['p95_ms']:.1f}"
        m[f"Lat{tag}PNinetyNine"] = f"{e['p99_ms']:.1f}"
        m[f"Lat{tag}Max"] = f"{e['max_ms']:.1f}"
        m[f"Lat{tag}Miss"] = pct(run["deadline_miss_rate"], 0 if run["deadline_miss_rate"] in (0.0, 1.0) else 2)
    m["LatPolicyPNinetyNine"] = f"{gs['stages']['policy']['p99_ms']:.2f}"
    m["LatVisionMean"] = f"{gs['stages']['vision']['mean_ms']:.1f}"
    m["LatVisionMax"] = f"{gs['stages']['vision']['max_ms']:.1f}"
    m["LatDecodeMax"] = f"{gs['stages']['decode']['max_ms']:.1f}"
    m["LatSpoolMax"] = f"{gs['stages']['spool']['max_ms']:.0f}"
    m["LatAuditMax"] = f"{gs['stages']['audit']['max_ms']:.0f}"
    m["LatPersistMax"] = f"{ga['stages']['persist']['max_ms']:.2f}"
    w = need(ga, "writer")
    m["WriterLagPNinetyNine"] = f"{w['lag']['p99_ms']:.1f}"
    m["WriterLagMax"] = f"{w['lag']['max_ms']:.1f}"
    m["WriterMaxDepth"] = str(w["max_queue_depth"])
    m["WriterErrors"] = str(w["errors"])
    m["LatNativeRes"] = f"${gs['native_frame_shape'][1]}\\times{gs['native_frame_shape'][0]}$"
    m["MemoryBankSize"] = fmt(gs["memory_bank_size"], 0)

    cases = {c["case"]: c for c in need(spool, "cases")}
    m["SpoolMaxOutage"] = str(int(max(c.get("outage_seconds", 0) for c in cases.values())))
    m["SpoolMaxEvents"] = fmt(max(c["events_generated"] for c in cases.values()), 0)
    m["SpoolUnexplainedMissing"] = str(sum(max(0, c["missing_events"] - c["evicted_by_capacity"]) for c in cases.values()))
    m["SpoolCrashDuplicates"] = str(cases["publisher_crash_sigkill"]["duplicate_deliveries"])
    ov = cases["overflow_60s_capacity_1000"]
    m["SpoolOverflowEvicted"] = fmt(ov["evicted_by_capacity"], 0)
    m["SpoolOverflowMissing"] = fmt(ov["missing_events"], 0)
    m["SpoolOverflowEvictedDelivered"] = str(max(0, ov["evicted_by_capacity"] - ov["missing_events"]))
    m["SpoolMaxDrain"] = f"{max(c['drain_seconds'] for c in cases.values()):.1f}"
    m["SpoolBrokerVersion"] = spool["broker"].split()[-1]

    # Subscriber crashes, inspection-process crashes with and without the writer thread, FULL cost.
    subs = {c["case"]: c for c in need(dfail, "subscriber_crash")}
    for key, tag in (("clean_session_10s", "Clean"), ("persistent_10s_default_queue", "PersistShort"),
                     ("persistent_60s_default_queue", "PersistLongDefault"),
                     ("persistent_60s_unlimited_queue", "PersistLongUnlimited")):
        c = subs[key]
        m[f"Sub{tag}Down"] = f"{c['subscriber_down_seconds']:.0f}"
        m[f"Sub{tag}WhileDown"] = fmt(c["events_generated_while_down"], 0)
        m[f"Sub{tag}Missing"] = fmt(c["missing_events"], 0)
    crash = {c["mode"]: c for c in need(dfail, "inspection_crash")}
    for key, tag in (("synchronous", "Sync"), ("writer_thread", "Writer")):
        c = crash[key]
        m[f"Crash{tag}Kills"] = str(c["kills"])
        m[f"Crash{tag}HandedOff"] = fmt(c["decisions_handed_off"], 0)
        m[f"Crash{tag}Lost"] = str(c["decisions_lost"])
        m[f"Crash{tag}KillsWithLoss"] = str(c["kills_with_loss"])
        m[f"Crash{tag}MaxLost"] = str(c["max_lost_per_kill"])
    cost = {r["synchronous"]: r for r in need(dfail, "spool_insert_cost")}
    for key, tag in (("NORMAL", "Normal"), ("FULL", "Full")):
        m[f"Insert{tag}Median"] = f"{cost[key]['median_ms']:.2f}"
        m[f"Insert{tag}PNinetyNine"] = f"{cost[key]['p99_ms']:.2f}"
        m[f"Insert{tag}Max"] = f"{cost[key]['max_ms']:.1f}"
    m["InsertCount"] = fmt(cost["FULL"]["inserts"], 0)

    # Held-out VisA evaluation (policy and parameters frozen) and its correlated rerun.
    m["VisaNumCategories"] = str(len(need(visa_bank, "categories")))
    va = [c["image_auroc"] for c in visa_bank["categories"]]
    m["VisaAurocMin"], m["VisaAurocMax"] = f"{min(va):.3f}", f"{max(va):.3f}"
    m["VisaGlareMedianMin"] = f"{min(c['median_norm_score_glare'] for c in visa_bank['categories']):.2f}"
    m["VisaDefectMedianBelowHigh"] = str(sum(c["median_norm_score_defect"] < 0.8 for c in visa_bank["categories"]))
    m["VisaDefectMedianBelowMed"] = str(sum(c["median_norm_score_defect"] < 0.5 for c in visa_bank["categories"]))
    m["VisaReAlertsFull"] = fmt(g("sustained_defects", "FULL_POLICY", "re_alerts_per_episode", units_visa), 2)
    m["VisaTotalRuns"] = fmt(len(list((RES / VISA_DIR).glob("*/*_seed*.json"))), 0)
    m["VisaUnits"] = str(len(units_visa[("nominal", "FULL_POLICY")]))
    for pol, tag in (("BASELINE", "Baseline"), ("EMA_KOFN", "EmaKofn"), ("FULL_POLICY", "Full"),
                     ("DELAY_TIMER", "Timer"), ("EMA_HYSTERESIS", "Hyst"), ("DECISION_FUSION", "Fusion"),
                     ("NO_DIVERGENCE", "NoDiv"), ("NO_STATE_GATING", "NoGating"), ("NO_FUSION", "NoFusion")):
        for u, pre in ((units_visa, "Visa"), (units_visa_rho, "VisaRho")):
            m[f"{pre}NominalFA{tag}"] = fmt(g("nominal", pol, "false_alerts_per_hour", u), 1)
            m[f"{pre}Glare{tag}"] = fmt(g("transient_glitches", pol, "ev:per_glare", u), 2)
            m[f"{pre}AlertsEp{tag}"] = fmt(g("sustained_defects", pol, "ev:per_episode", u), 2)
            m[f"{pre}Delay{tag}"] = fmt(g("sustained_defects", pol, "mean_delay_frames", u), 1)
            m[f"{pre}Recall{tag}"] = fmt(g("multimodal_faults", pol, "routing_recall", u), 2)
            m[f"{pre}DefectRecall{tag}"] = fmt(g("sustained_defects", pol, "routing_recall", u), 2)
            m[f"{pre}CritFA{tag}"] = fmt(g("multimodal_faults", pol, "false_high_alerts_per_hour", u), 1)
            m[f"{pre}DegradedFA{tag}"] = fmt(g("degraded_inputs", pol, "false_alerts_per_hour", u), 1)
            m[f"{pre}StateFA{tag}"] = fmt(g("state_transitions", pol, "false_alerts_per_hour", u), 1)
    for (metric, wl, comp), r in effects_visa.items():
        key = {"false_alerts_per_hour": "FA", "false_high_alerts_per_hour": "CritFA", "ev:per_glare": "Glare",
               "ev:per_episode": "AlertsEp", "mean_delay_frames": "Delay", "routing_recall": "Recall"}[metric]
        wtag = {"nominal": "Nom", "transient_glitches": "", "sustained_defects": "", "multimodal_faults": "Multi",
                "degraded_inputs": "Degr", "state_transitions": "State"}[wl]
        ctag = {"DELAY_TIMER": "Timer", "EMA_HYSTERESIS": "Hyst", "DECISION_FUSION": "Fusion", "BASELINE": "Baseline"}[comp]
        nd = 2 if metric in ("ev:per_glare", "ev:per_episode", "routing_recall") else 1
        base = f"VisaEff{key}{wtag}{ctag}"
        m[base], m[base + "Lo"], m[base + "Hi"] = fmt(r["mean"], nd), fmt(r["lo"], nd), fmt(r["hi"], nd)
        m[base + "Wins"], m[base + "Losses"] = str(r["ref_better_categories"]), str(r["ref_worse_categories"])
        m[base + "P"] = fmt_p(r["perm_p"])
    m["PermMinPMvtec"] = f"{2 / 2 ** len(CATEGORIES):.3f}"
    m["PermMinPVisa"] = f"{2 / 2 ** len(visa_bank['categories']):.4f}"

    t = need(tr, "traces")
    full_first = [t[k]["policies"]["FULL_POLICY"]["first_flag_rel_onset"]["mean"] for k in t]
    m["TraceFirstFlagFullMin"] = fmt(-max(full_first), 0)
    m["TraceFirstFlagFullMax"] = fmt(-min(full_first), 0)
    m["TraceAlertsFromOnsetFull"] = fmt(max(t[k]["policies"]["FULL_POLICY"]["alerts_from_onset"]["mean"] for k in t), 0)
    return m


def main() -> None:
    summary = load(RES / "ablation" / "ablation_summary.json")
    sens = load(RES / "sensitivity" / "sensitivity_summary.json")
    lat = load(RES / "latency_benchmark_summary.json")
    spool = load(RES / "spooler_stress" / "spooler_stress_summary.json")
    dfail = load(RES / "spooler_stress" / "delivery_failures_summary.json")
    tr = load(RES / "real_trace_benchmark_summary.json")
    bank = load(RES / "score_bank" / "summary.json")
    optical = load(RES / "score_bank" / "optical_check_on_test.json")
    short = load(RES / "short_defect_recall.json")

    CATS_BY_DIR.update(dict(enumerate(CATEGORIES)))
    units = load_units("ablation")
    units_rho = load_units(RHO_DIR)
    visa_bank = load(RES / "score_bank_visa" / "summary.json")
    units_visa = load_units(VISA_DIR)
    units_visa_rho = load_units(VISA_RHO_DIR)
    ev_out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for (wl, pol), cell in units.items():
        for key in ("ev:per_glare", "ev:per_episode"):
            vals = {u: m[key] for u, m in cell.items() if key in m}
            if vals:
                ev_out.setdefault(wl, {}).setdefault(pol, {})[key[3:]] = {
                    **cluster_boot(vals), "per_unit": {str(u): v for u, v in sorted(vals.items())}}
    (RES / "paper_event_metrics.json").write_text(json.dumps(ev_out, indent=1), encoding="utf-8")

    (PAPER / "tables").mkdir(parents=True, exist_ok=True)
    (PAPER / "figures").mkdir(parents=True, exist_ok=True)
    tables = {
        "ablation": table_ablation(units, POLICIES),
        "ablation_rho": table_ablation(units_rho, ["BASELINE", "EMA_KOFN", "FULL_POLICY", "DELAY_TIMER",
                                                    "EMA_HYSTERESIS", "DECISION_FUSION"]),
        "per_category": table_per_category(units),
        "durability": table_durability(spool),
        "delivery_failures": table_delivery_failures(dfail),
        "latency": table_latency(lat),
        "trace": table_trace(tr),
    }
    eff_tex, effects = table_effects(units, ["BASELINE"] + EXTERNAL)
    tables["effects"] = eff_tex
    tables["ablation_visa"] = table_ablation(units_visa, POLICIES)
    CATS_BY_DIR.clear()
    CATS_BY_DIR.update({i: c["category"] for i, c in enumerate(visa_bank["categories"])})
    eff_visa_tex, effects_visa = table_effects(units_visa, ["BASELINE"] + EXTERNAL)
    tables["effects_visa"] = eff_visa_tex
    CATS_BY_DIR.clear()
    CATS_BY_DIR.update(dict(enumerate(CATEGORIES)))
    rows = [{"dataset": ds, "metric": mt, "workload": wl, "comparator": c, "category": cat, "delta_comparator_minus_full": v,
             "perm_p": r["perm_p"]}
            for ds, eff in (("mvtec", effects), ("visa", effects_visa)) for (mt, wl, c), r in eff.items()
            for cat, v in r["category_deltas"].items()]
    import csv
    with open(RES / "paper_category_deltas.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for name, tex in tables.items():
        (PAPER / "tables" / f"{name}.tex").write_text(tex + "\n", encoding="utf-8")

    plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    crossings = fig_operator_load(units, PAPER / "figures" / "operator_load.pdf")
    fig_sensitivity(sens, PAPER / "figures" / "sensitivity.pdf")
    fig_category_deltas(effects, PAPER / "figures" / "category_deltas.pdf")

    mac = macros(units, units_rho, effects, summary, sens, lat, spool, tr, bank, optical, crossings, short,
                 units_visa, units_visa_rho, effects_visa, visa_bank, dfail)
    lines = ["% Generated by scripts/build_paper_assets.py from results/*.json. Do not edit by hand."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in mac.items()]
    (PAPER / "generated_metrics.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(mac)} macros, {len(tables)} tables, 3 figures")
    for k, v in mac.items():
        print(f"  {k:32s} {v}")


if __name__ == "__main__":
    main()
