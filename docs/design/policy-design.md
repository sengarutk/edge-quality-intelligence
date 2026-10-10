# Decision Policy

Implementation: `src/runtime/policy.py`; parameters: `configs/policy_config.yaml`
(meanings in `configs/threshold_manifest.json`); contract tests: `tests/test_rt_policy.py`.

1. **Smoothing.** `v_ema = a_v v + (1 - a_v) v_ema` (a_v = 0.35) and likewise for the sensor
   score (a_s = 0.25). Frames that fail the optical check do not update `v_ema`; readings with a
   missing channel do not update `s_ema`.
2. **Persistence (k-of-N).** Visual evidence is *high* if at least k = 4 of the last N = 10
   values of `v_ema` are >= 0.8, *medium* for the same test at 0.5; sensor evidence is confirmed
   when k of the last N values of `s_ema` are >= 0.7.
3. **Candidate decision** (first matching rule):
   machine FAULT -> HIGH; camera degraded -> REVIEW if sensors confirm, REVIEW for camera
   maintenance if k of the last N frames were degraded, otherwise held; high visual evidence +
   sensor confirmation -> HIGH; high visual evidence without confirmed sensor
   evidence -> REVIEW if a channel is missing (sensor fallback), or if |v_ema - s_ema| >= 0.45 or the
   current v_ema is below 0.8 (cross-modal discrepancy), HIGH otherwise (always HIGH in the
   NO_DIVERGENCE / NO_FUSION ablations, apart from the sensor fallback in NO_DIVERGENCE); sensor
   confirmation alone or medium visual evidence -> REVIEW.

   The "current v_ema below 0.8" condition was added after the first full ablation: without it the
   k-of-N window still reported high evidence for several frames after a glare burst had ended, while
   |v_ema - s_ema| had already fallen below 0.45, so the decaying glare escalated to HIGH on nominal
   sensors (test: `test_glare_tail_does_not_escalate_to_high`). All reported results use the revised rule.
4. **State gating.** In IDLE or MAINTENANCE the candidate is lowered one level (HIGH -> REVIEW,
   REVIEW -> NORMAL).
5. **Incident latch.** A non-normal decision opens an incident (one alert) or joins the open one
   silently; a severity increase inside an incident emits one more alert. The incident closes after
   T_cool = 15 consecutive normal frames. `is_new_alert` marks the decisions that enter the
   operator queue.

Ablation modes: BASELINE (raw score thresholds, per-frame alerts), EMA_ONLY, EMA_KOFN (vision
only, no latch), NO_COOLDOWN (no latch), NO_FUSION, NO_DIVERGENCE, NO_STATE_GATING, FULL_POLICY.

External baselines (`src/runtime/alarm_baselines.py`, same thresholds, no tuning): DELAY_TIMER
(on-delay of k consecutive raw samples, off-delay of T_cool samples), EMA_HYSTERESIS (EMA with a 0.1
deadband) and DECISION_FUSION (HIGH when a vision delay-timer alarm and a sensor delay-timer alarm
are both active, REVIEW when one is). Each emits an alert when its alarm severity rises.
