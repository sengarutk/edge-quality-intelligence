#!/usr/bin/env python3
"""Build every number, table and figure used by paper/main.tex from result files.

Inputs (all required; a missing file or key is an error, never a default):
  results/ablation/<workload>/<POLICY>_seed<unit>.json   (scripts/run_ablation_study.py)
  results/ablation/ablation_summary.json
  results/sensitivity/sensitivity_summary.json           (scripts/run_sensitivity_analysis.py)
  results/latency_benchmark_summary.json                 (scripts/benchmark_latency.py)
  results/spooler_stress/spooler_stress_summary.json     (scripts/benchmark_spooler_resilience.py)
  results/real_trace_benchmark_summary.json              (scripts/run_real_trace_benchmark.py)
  results/score_bank/summary.json, optical_check_on_test.json

Outputs:
  paper/generated_metrics.tex      LaTeX macros (\\newcommand)
  paper/tables/*.tex               tables included by main.tex
  paper/figures/*.pdf              figures included by main.tex
  results/paper_event_metrics.json      event-normalized metrics computed here
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.experiments.workloads import build_timeline, load_workload  # noqa: E402
from src.metrics.stats import bootstrap_ci  # noqa: E402

RES = ROOT / "results"
PAPER = ROOT / "paper"
POLICIES = ["BASELINE", "EMA_ONLY", "EMA_KOFN", "NO_COOLDOWN", "NO_FUSION", "NO_DIVERGENCE", "NO_STATE_GATING",
            "FULL_POLICY"]
GRACE = 15
MU = 60.0  # reviews per hour, one operator (about one minute per review)
# Validated categorical slots (light surface) + distinct markers as secondary encoding.
STYLE = {"BASELINE": ("#2a78d6", "o"), "EMA_KOFN": ("#eb6834", "s"), "NO_DIVERGENCE": ("#1baf7a", "^"),
         "FULL_POLICY": ("#eda100", "D")}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


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


def runs(extra: bool = True):
    for f in sorted((RES / "ablation").glob("*/*_seed*.json")):
        yield json.loads(f.read_text(encoding="utf-8"))


def spans(mask: np.ndarray):
    idx = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return list(zip(idx[::2], idx[1::2]))


def event_metrics() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Per (workload, policy): alerts per defect episode and per glare burst, averaged over units."""
    wl_cache, tl_cache = {}, {}
    acc: Dict[tuple, Dict[str, List[float]]] = {}
    for r in runs():
        wl = r["workload"]
        if wl not in wl_cache:
            wl_cache[wl] = load_workload(ROOT / "configs" / "scenarios" / f"{wl}.yaml")
        key = (wl, r["seed"])
        if key not in tl_cache:
            tl_cache[key] = build_timeline(wl_cache[wl], r["seed"])
        tl = tl_cache[key]
        covered = np.zeros(tl.n, bool)
        for a, b in spans(tl.actionable):
            covered[a:min(tl.n, b + GRACE)] = True
        alerts = np.zeros(tl.n, bool)
        alerts[r["alert_steps"]] = True
        d = acc.setdefault((wl, r["policy_mode"]), {"per_glare": [], "per_episode": []})
        bursts = [(a, b) for a, b in spans(tl.vision_source == 2) if tl.machine_state[a] == "RUNNING"]
        if bursts:
            n = sum(int(alerts[a:min(tl.n, b + GRACE)][~covered[a:min(tl.n, b + GRACE)]].sum()) for a, b in bursts)
            d["per_glare"].append(n / len(bursts))
        eps = spans(tl.actionable)
        if eps:
            d["per_episode"].append(float(np.mean([alerts[a:min(tl.n, b + GRACE)].sum() for a, b in eps])))
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for (wl, pol), d in acc.items():
        out.setdefault(wl, {})[pol] = {k: dict(bootstrap_ci(v, n_boot=2000)) for k, v in d.items() if v}
    return out


def mean(summary, wl, pol, metric):
    return need(summary, "scenarios", wl, pol, metric, "mean")


def fmt(x: float, nd: int = 1) -> str:
    if x is None:
        return "--"
    if abs(x) >= 1000:
        return f"{x:,.0f}".replace(",", "{,}")
    return f"{x:.{nd}f}"


def pct(x: float, nd: int = 1) -> str:
    return f"{100 * x:.{nd}f}\\%"


# ----------------------------------------------------------------- tables
def table_ablation(summary, ev) -> str:
    rows = []
    for pol in POLICIES:
        cells = [
            fmt(mean(summary, "nominal", pol, "false_alerts_per_hour")),
            fmt(need(ev, "transient_glitches", pol, "per_glare", "mean"), 2),
            fmt(need(ev, "sustained_defects", pol, "per_episode", "mean"), 2),
            fmt(mean(summary, "sustained_defects", pol, "mean_delay_frames")),
            fmt(mean(summary, "multimodal_faults", pol, "routing_recall"), 2),
            fmt(mean(summary, "multimodal_faults", pol, "false_high_alerts_per_hour")),
            fmt(mean(summary, "degraded_inputs", pol, "false_alerts_per_hour")),
            fmt(mean(summary, "state_transitions", pol, "false_alerts_per_hour")),
        ]
        name = pol.replace("_", "\\_")
        rows.append(f"\\texttt{{{name}}} & " + " & ".join(cells) + " \\\\")
    return "\n".join([
        "\\begin{tabular}{@{}lrrrrrrrr@{}}", "\\toprule",
        " & Nominal & Glare & \\multicolumn{2}{c}{Sustained defects} & \\multicolumn{2}{c}{Multi-modal faults} & Degraded & States \\\\",
        "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
        "Policy & FA/h & alerts/burst & alerts/ep. & delay (fr.) & recall & crit.\\ FA/h & FA/h & FA/h \\\\",
        "\\midrule", *rows, "\\bottomrule", "\\end{tabular}",
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


def table_latency(lat) -> str:
    runs_ = {r["device"]: r for r in need(lat, "runs")}
    gpu, cpu = runs_["cuda"], runs_["cpu"]
    labels = {"decode": "PNG decode", "vision": "Optical check + PatchCore", "sensor": "Sensor step",
              "policy": "Policy", "spool": "Spool insert", "audit": "Audit insert", "e2e": "End to end"}
    rows = []
    for k, lab in labels.items():
        g, c = gpu["stages"][k], cpu["stages"][k]
        sep = "\\midrule\n" if k == "e2e" else ""
        rows.append(f"{sep}{lab} & {g['mean_ms']:.2f} & {g['p99_ms']:.2f} & {g['max_ms']:.1f} & "
                    f"{c['mean_ms']:.2f} & {c['p99_ms']:.2f} \\\\")
    return "\n".join(["\\begin{tabular}{@{}lrrrrr@{}}", "\\toprule",
                      " & \\multicolumn{3}{c}{GPU} & \\multicolumn{2}{c}{CPU} \\\\",
                      "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}",
                      "Stage (ms) & mean & p99 & max & mean & p99 \\\\", "\\midrule", *rows,
                      "\\bottomrule", "\\end{tabular}"])


def table_trace(tr) -> str:
    rows = []
    names = {"bearing_proxy": "Bearing proxy", "thermal_creep_proxy": "Thermal-creep proxy"}
    for t, info in need(tr, "traces").items():
        for pol in ("BASELINE", "NO_FUSION", "FULL_POLICY"):
            s = need(info, "policies", pol)
            delay = s.get("mean_delay_frames", {}).get("mean")
            rows.append(f"{names[t] if pol == 'BASELINE' else ''} & \\texttt{{{pol.replace('_', chr(92) + '_')}}} & "
                        f"{fmt(need(s, 'false_alerts_per_hour', 'mean'), 0)} & "
                        f"{fmt(need(s, 'routing_recall', 'mean'), 2)} & {fmt(delay)} \\\\")
        rows.append("\\midrule")
    rows[-1] = "\\bottomrule"
    return "\n".join(["\\begin{tabular}{@{}llrrr@{}}", "\\toprule",
                      "Trace & Policy & pre-fault FA/h & recall & delay (fr.) \\\\", "\\midrule", *rows,
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


def fig_operator_load(summary, ev, path: Path, glare_per_hour: float) -> Dict[str, float]:
    """rho(r) = [r * alerts/episode + g * alerts/glare burst + nominal FA/h] / mu."""
    r = np.linspace(0, 30, 121)
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    crossings = {}
    for pol, (color, marker) in STYLE.items():
        lam = (r * need(ev, "sustained_defects", pol, "per_episode", "mean")
               + glare_per_hour * need(ev, "transient_glitches", pol, "per_glare", "mean")
               + mean(summary, "nominal", pol, "false_alerts_per_hour"))
        rho = lam / MU
        ax.plot(r, rho, color=color, linewidth=2, marker=marker, markevery=(5 * list(STYLE).index(pol), 20), markersize=5,
                label=pol.replace("_", " ").lower())
        # Largest episode rate with rho < 1 (0 if the queue is unstable even without defects).
        a_ep = need(ev, "sustained_defects", pol, "per_episode", "mean")
        spare = MU - glare_per_hour * need(ev, "transient_glitches", pol, "per_glare", "mean") \
            - mean(summary, "nominal", pol, "false_alerts_per_hour")
        crossings[pol] = max(0.0, spare / a_ep)
    ax.axhline(1.0, color=INK, linewidth=0.8, linestyle="--")
    ax.text(29.5, 1.1, "$\\rho = 1$", ha="right", va="bottom", fontsize=7, color=INK)
    ax.set_yscale("log")
    ax.set_xlabel("Defect episodes per hour", fontsize=8, color=INK)
    ax.set_ylabel("Reviewer utilization $\\rho$", fontsize=8, color=INK)
    style_axes(ax)
    ax.legend(fontsize=6.5, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.22))
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return crossings


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
def macros(summary, ev, sens, lat, spool, tr, bank, optical, crossings, glare_rate) -> Dict[str, str]:
    m: Dict[str, str] = {}
    design = need(summary, "design")
    m["NumPolicies"] = str(len(design["policy_modes"]))
    m["NumWorkloads"] = str(len(design["workloads"]))
    m["NumCategories"] = str(len(design["categories"]))
    m["NumSeeds"] = str(len(design["seeds"]))
    m["UnitsPerCell"] = str(design["units_per_cell"])
    m["StepsPerRun"] = fmt(design["steps_per_run"], 0)
    m["TotalRuns"] = fmt(len(list((RES / "ablation").glob("*/*_seed*.json"))), 0)

    aurocs = [c["image_auroc"] for c in need(bank, "categories")]
    m["BankAurocMin"], m["BankAurocMax"] = f"{min(aurocs):.3f}", f"{max(aurocs):.3f}"
    med_def = [c["median_norm_score_defect"] for c in bank["categories"]]
    m["DefectMedianBelowHigh"] = str(sum(v < 0.8 for v in med_def))
    m["GlareMedianMin"] = f"{min(c['median_norm_score_glare'] for c in bank['categories']):.2f}"
    worst = max(optical.items(), key=lambda kv: kv[1]["flag_rate_defective"])
    m["OpticalWorstCategory"] = worst[0].replace("_", "\\_")
    m["OpticalWorstRate"] = pct(worst[1]["flag_rate_defective"])
    m["OpticalWorstTypes"] = ", ".join(t.replace("_", "\\_") for t in worst[1]["flagged_by_defect_type"])
    m["OpticalNominalFlagMax"] = pct(max(v["flag_rate_nominal"] for v in optical.values()))

    g = lambda wl, pol, k: mean(summary, wl, pol, k)  # noqa: E731
    m["NominalFABaseline"] = fmt(g("nominal", "BASELINE", "false_alerts_per_hour"), 0)
    m["NominalFAFull"] = fmt(g("nominal", "FULL_POLICY", "false_alerts_per_hour"), 1)
    for pol, tag in (("BASELINE", "Baseline"), ("EMA_KOFN", "EmaKofn"), ("FULL_POLICY", "Full")):
        m[f"GlarePerBurst{tag}"] = fmt(need(ev, "transient_glitches", pol, "per_glare", "mean"), 2)
        m[f"AlertsPerEpisode{tag}"] = fmt(need(ev, "sustained_defects", pol, "per_episode", "mean"), 2)
    m["SustainedAlertsBaseline"] = fmt(g("sustained_defects", "BASELINE", "alerts_per_hour"), 0)
    m["SustainedAlertsFull"] = fmt(g("sustained_defects", "FULL_POLICY", "alerts_per_hour"), 0)
    red = 1 - g("sustained_defects", "FULL_POLICY", "alerts_per_hour") / g("sustained_defects", "BASELINE", "alerts_per_hour")
    m["SustainedAlertReduction"] = pct(red)
    m["SustainedDelayFull"] = fmt(g("sustained_defects", "FULL_POLICY", "mean_delay_frames"))
    m["SustainedDelayBaseline"] = fmt(g("sustained_defects", "BASELINE", "mean_delay_frames"), 2)
    m["MultiRecallFull"] = fmt(g("multimodal_faults", "FULL_POLICY", "routing_recall"), 2)
    m["MultiRecallNoFusion"] = fmt(g("multimodal_faults", "NO_FUSION", "routing_recall"), 2)
    m["MultiRecallBaseline"] = fmt(g("multimodal_faults", "BASELINE", "routing_recall"), 2)
    m["CritFAFull"] = fmt(g("multimodal_faults", "FULL_POLICY", "false_high_alerts_per_hour"))
    m["CritFANoDiv"] = fmt(g("multimodal_faults", "NO_DIVERGENCE", "false_high_alerts_per_hour"))
    m["CritFABaseline"] = fmt(g("multimodal_faults", "BASELINE", "false_high_alerts_per_hour"), 0)
    m["StateFAFull"] = fmt(g("state_transitions", "FULL_POLICY", "false_alerts_per_hour"))
    m["StateFANoGating"] = fmt(g("state_transitions", "NO_STATE_GATING", "false_alerts_per_hour"))
    m["DegradedFAFull"] = fmt(g("degraded_inputs", "FULL_POLICY", "false_alerts_per_hour"))
    reasons: Dict[str, int] = {}
    for r in runs():
        if r["workload"] == "degraded_inputs" and r["policy_mode"] == "FULL_POLICY":
            for k, v in r["alert_reasons"].items():
                reasons[k] = reasons.get(k, 0) + v
    m["DegradedAlertsTotal"] = str(sum(reasons.values()))
    m["DegradedOpticalAlerts"] = str(reasons.get("OPTICAL_DEGRADATION_FALLBACK", 0))
    m["DegradedSensorAlerts"] = str(reasons.get("SUSTAINED_SENSOR_ANOMALY", 0))
    m["DegradedFANoCooldown"] = fmt(g("degraded_inputs", "NO_COOLDOWN", "false_alerts_per_hour"), 0)

    # Holm-adjusted p-values of FULL_POLICY vs BASELINE on alerts per hour, worst case over workloads.
    ps = []
    for wl, pols in need(summary, "scenarios").items():
        for t in pols.get("FULL_POLICY", {}).get("significance_vs_baseline", []):
            if t["metric_name"] == "alerts_per_hour":
                ps.append(t["adjusted_p_value"])
    m["FullVsBaselineMaxP"] = f"{max(ps):.1e}".replace("e-0", "\\times10^{-").replace("e-", "\\times10^{-") + "}"

    # sensitivity endpoints (multi-modal, critical false alarms) for k
    kpts = need(sens, "sweeps", "required_k", "multimodal_faults")
    m["SensKLow"], m["SensKHigh"] = str(int(kpts[0]["value"])), str(int(kpts[-1]["value"]))
    m["SensCritKLow"] = fmt(kpts[0]["false_high_alerts_per_hour"]["mean"])
    m["SensCritKHigh"] = fmt(kpts[-1]["false_high_alerts_per_hour"]["mean"])
    dpts = need(sens, "sweeps", "required_k", "sustained_defects")
    m["SensDelayKLow"] = fmt(dpts[0]["mean_delay_frames"]["mean"])
    m["SensDelayKHigh"] = fmt(dpts[-1]["mean_delay_frames"]["mean"])
    rec = [e["routing_recall"]["mean"] for p in sens["sweeps"].values() for e in p["multimodal_faults"]]
    m["SensRecallMin"] = fmt(min(rec), 2)

    m["GlareRateAssumed"] = fmt(glare_rate, 0)
    m["MuReviews"] = fmt(MU, 0)
    m["LoadCrossFull"] = fmt(crossings["FULL_POLICY"], 0)
    m["LoadCrossEmaKofn"] = fmt(crossings["EMA_KOFN"], 0)

    runs_ = {r["device"]: r for r in need(lat, "runs")}
    gpu, cpu = runs_["cuda"], runs_["cpu"]
    m["LatGpuName"] = gpu["device_name"].replace("NVIDIA GeForce ", "")
    m["LatCycles"] = fmt(gpu["cycles"], 0)
    m["LatCpuCycles"] = fmt(cpu["cycles"], 0)
    m["LatGpuMean"] = f"{gpu['stages']['e2e']['mean_ms']:.1f}"
    m["LatGpuPNinetyFive"] = f"{gpu['stages']['e2e']['p95_ms']:.1f}"
    m["LatGpuPNinetyNine"] = f"{gpu['stages']['e2e']['p99_ms']:.1f}"
    m["LatGpuMax"] = f"{gpu['stages']['e2e']['max_ms']:.1f}"
    m["LatGpuMiss"] = pct(gpu["deadline_miss_rate"], 2)
    m["LatCpuMean"] = f"{cpu['stages']['e2e']['mean_ms']:.1f}"
    m["LatCpuMiss"] = pct(cpu["deadline_miss_rate"], 0)
    m["LatPolicyPNinetyNine"] = f"{gpu['stages']['policy']['p99_ms']:.2f}"
    m["LatVisionMean"] = f"{gpu['stages']['vision']['mean_ms']:.1f}"
    m["LatSpoolMax"] = f"{gpu['stages']['spool']['max_ms']:.0f}"
    m["LatAuditMax"] = f"{gpu['stages']['audit']['max_ms']:.0f}"
    m["LatNativeRes"] = f"${gpu['native_frame_shape'][1]}\\times{gpu['native_frame_shape'][0]}$"
    m["MemoryBankSize"] = fmt(gpu["memory_bank_size"], 0)

    cases = {c["case"]: c for c in need(spool, "cases")}
    m["SpoolMaxOutage"] = str(int(max(c.get("outage_seconds", 0) for c in cases.values())))
    m["SpoolMaxEvents"] = fmt(max(c["events_generated"] for c in cases.values()), 0)
    m["SpoolUnexplainedMissing"] = str(sum(c["unexplained_missing"] for c in cases.values()))
    m["SpoolCrashDuplicates"] = str(cases["publisher_crash_sigkill"]["duplicate_deliveries"])
    m["SpoolOverflowEvicted"] = fmt(cases["overflow_60s_capacity_1000"]["evicted_by_capacity"], 0)
    m["SpoolMaxDrain"] = f"{max(c['drain_seconds'] for c in cases.values()):.1f}"
    m["SpoolBrokerVersion"] = spool["broker"].split()[-1] if "broker" in spool else "?"

    t = need(tr, "traces")
    m["TraceRecallFull"] = fmt(min(t[k]["policies"]["FULL_POLICY"]["routing_recall"]["mean"] for k in t), 2)
    m["TraceRecallBaselineMax"] = fmt(max(t[k]["policies"]["BASELINE"]["routing_recall"]["mean"] for k in t), 2)
    try:
        m["GitCommit"] = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        m["GitCommit"] = "unknown"
    return m


def main() -> None:
    summary = load(RES / "ablation" / "ablation_summary.json")
    sens = load(RES / "sensitivity" / "sensitivity_summary.json")
    lat = load(RES / "latency_benchmark_summary.json")
    spool = load(RES / "spooler_stress" / "spooler_stress_summary.json")
    tr = load(RES / "real_trace_benchmark_summary.json")
    bank = load(RES / "score_bank" / "summary.json")
    optical = load(RES / "score_bank" / "optical_check_on_test.json")

    ev = event_metrics()
    (RES / "paper_event_metrics.json").write_text(json.dumps(ev, indent=1), encoding="utf-8")

    (PAPER / "tables").mkdir(parents=True, exist_ok=True)
    (PAPER / "figures").mkdir(parents=True, exist_ok=True)
    (PAPER / "tables" / "ablation.tex").write_text(table_ablation(summary, ev) + "\n", encoding="utf-8")
    (PAPER / "tables" / "durability.tex").write_text(table_durability(spool) + "\n", encoding="utf-8")
    (PAPER / "tables" / "latency.tex").write_text(table_latency(lat) + "\n", encoding="utf-8")
    (PAPER / "tables" / "trace.tex").write_text(table_trace(tr) + "\n", encoding="utf-8")

    plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    glare_rate = 30.0
    crossings = fig_operator_load(summary, ev, PAPER / "figures" / "operator_load.pdf", glare_rate)
    fig_sensitivity(sens, PAPER / "figures" / "sensitivity.pdf")

    mac = macros(summary, ev, sens, lat, spool, tr, bank, optical, crossings, glare_rate)
    lines = ["% Generated by scripts/build_paper_assets.py from results/*.json. Do not edit by hand."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in mac.items()]
    (PAPER / "generated_metrics.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(mac)} macros, 4 tables, 2 figures")
    for k, v in mac.items():
        print(f"  {k:28s} {v}")


if __name__ == "__main__":
    main()
