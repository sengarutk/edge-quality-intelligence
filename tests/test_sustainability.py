import pytest
from src.sustainability.qcf_engine import QualityCarbonFootprintEngine, SustainabilityParameters
from src.sustainability.sqi_calculator import SustainabilityQualityIndexCalculator


def test_qcf_engine_initialization_and_configs():
    # Test loading from YAML
    params_nut = SustainabilityParameters.from_yaml(part_name="metal_nut")
    assert params_nut.m_part == 0.45
    assert params_nut.kappa_mat == 2.89
    assert params_nut.E_rework == 1.85
    assert params_nut.gamma_fatal == 0.35
    assert params_nut.theta_tier == 8.0

    params_tile = SustainabilityParameters.from_yaml(part_name="tile")
    assert params_tile.m_part == 1.20
    assert params_tile.kappa_mat == 0.78
    assert params_tile.E_rework == 3.20
    assert params_tile.gamma_fatal == 0.80
    assert params_tile.theta_tier == 4.5

    engine = QualityCarbonFootprintEngine(params=params_nut)
    assert engine.params.xi_grid == 0.230


def test_qcf_monotonicity_increasing_fn():
    """Increasing N_FN strictly increases QCF."""
    engine = QualityCarbonFootprintEngine(params=SustainabilityParameters.from_yaml(part_name="metal_nut"))
    
    # Baseline with 10 FN
    res_10 = engine.compute_annual_qcf(
        tp_count=80, fp_count=20, fn_count=10, tn_count=890, total_parts=1000
    )
    # Scaled with 30 FN
    res_30 = engine.compute_annual_qcf(
        tp_count=80, fp_count=20, fn_count=30, tn_count=870, total_parts=1000
    )
    # Scaled with 50 FN
    res_50 = engine.compute_annual_qcf(
        tp_count=80, fp_count=20, fn_count=50, tn_count=850, total_parts=1000
    )

    assert res_10["total_qcf_kg"] < res_30["total_qcf_kg"] < res_50["total_qcf_kg"]
    assert res_10["ghg_escape"] < res_30["ghg_escape"] < res_50["ghg_escape"]
    assert res_10["total_qcf_metric_tons"] < res_30["total_qcf_metric_tons"] < res_50["total_qcf_metric_tons"]


def test_sqi_monotonicity_increasing_fn():
    """Increasing N_FN strictly decreases SQI."""
    engine = QualityCarbonFootprintEngine(params=SustainabilityParameters.from_yaml(part_name="metal_nut"))
    calc = SustainabilityQualityIndexCalculator(qcf_engine=engine)

    baseline_counts = {"tp": 70, "fp": 50, "fn": 30, "tn": 850}
    base_qcf = engine.compute_annual_qcf(
        tp_count=70, fp_count=50, fn_count=30, tn_count=850, total_parts=1000
    )

    # Policy 1: low escapes (FN=5)
    counts_low_fn = {"tp": 95, "fp": 10, "fn": 5, "tn": 890}
    qcf_low_fn = engine.compute_annual_qcf(
        tp_count=95, fp_count=10, fn_count=5, tn_count=890, total_parts=1000
    )
    sqi_low_fn = calc.compute_sqi(qcf_low_fn, base_qcf, counts_low_fn, baseline_counts)

    # Policy 2: higher escapes (FN=25)
    counts_high_fn = {"tp": 75, "fp": 10, "fn": 25, "tn": 890}
    qcf_high_fn = engine.compute_annual_qcf(
        tp_count=75, fp_count=10, fn_count=25, tn_count=890, total_parts=1000
    )
    sqi_high_fn = calc.compute_sqi(qcf_high_fn, base_qcf, counts_high_fn, baseline_counts)

    assert sqi_low_fn["sqi"] > sqi_high_fn["sqi"]
    assert sqi_low_fn["csf"] > sqi_high_fn["csf"]


def test_perfect_detection_optimal_sqi():
    """Perfect detection (FN=0, FP=0) achieves optimal SQI over imperfect policies."""
    engine = QualityCarbonFootprintEngine(params=SustainabilityParameters.from_yaml(part_name="metal_nut"))
    calc = SustainabilityQualityIndexCalculator(qcf_engine=engine)

    baseline_counts = {"tp": 80, "fp": 80, "fn": 20, "tn": 820}
    base_qcf = engine.compute_annual_qcf(80, 80, 20, 820, 1000)

    imperfect_counts = {"tp": 90, "fp": 30, "fn": 10, "tn": 870}
    imp_qcf = engine.compute_annual_qcf(90, 30, 10, 870, 1000)
    sqi_imp = calc.compute_sqi(imp_qcf, base_qcf, imperfect_counts, baseline_counts)

    perfect_counts = {"tp": 100, "fp": 0, "fn": 0, "tn": 900}
    perf_qcf = engine.compute_annual_qcf(100, 0, 0, 900, 1000)
    sqi_perf = calc.compute_sqi(perf_qcf, base_qcf, perfect_counts, baseline_counts)

    assert sqi_perf["sqi"] > sqi_imp["sqi"]
    assert 0.0 <= sqi_perf["sqi"] <= 1.0
    assert 0.0 <= sqi_perf["msf"] <= 1.0
    assert 0.0 <= sqi_perf["esf"] <= 1.0
    assert 0.0 <= sqi_perf["csf"] <= 1.0
    assert 0.0 <= sqi_perf["cf"] <= 1.0


def test_components_non_negative_and_bounded():
    """All components of QCF and SQI are non-negative and bounded."""
    for part in ["metal_nut", "tile"]:
        engine = QualityCarbonFootprintEngine(params=SustainabilityParameters.from_yaml(part_name=part))
        calc = SustainabilityQualityIndexCalculator(qcf_engine=engine)

        qcf = engine.compute_annual_qcf(tp_count=85, fp_count=15, fn_count=15, tn_count=885, total_parts=1000)
        assert qcf["ghg_scrap"] >= 0.0
        assert qcf["ghg_rework"] >= 0.0
        assert qcf["ghg_escape"] >= 0.0
        assert qcf["ghg_compute"] >= 0.0
        assert qcf["total_qcf_kg"] >= 0.0
        assert qcf["total_qcf_metric_tons"] >= 0.0

        base_qcf = engine.compute_annual_qcf(tp_count=50, fp_count=50, fn_count=50, tn_count=850, total_parts=1000)
        sqi = calc.compute_sqi(
            qcf, base_qcf,
            {"tp": 85, "fp": 15, "fn": 15, "tn": 885},
            {"tp": 50, "fp": 50, "fn": 50, "tn": 850}
        )

        assert 0.0 <= sqi["msf"] <= 1.0
        assert 0.0 <= sqi["esf"] <= 1.0
        assert 0.0 <= sqi["csf"] <= 1.0
        assert 0.0 <= sqi["cf"] <= 1.0
        assert 0.0 <= sqi["sqi"] <= 1.0


def test_flagged_parts_are_scrapped_or_reworked_not_both():
    from src.sustainability.qcf_engine import QualityCarbonFootprintEngine, SustainabilityParameters

    p = SustainabilityParameters(gamma_false_scrap=0.2, xi_grid=1.0, E_rework=1.0, m_part=1.0, kappa_mat=1.0,
                                 P_edge=0.0, gamma_fatal=0.0)
    res = QualityCarbonFootprintEngine(params=p).compute_annual_qcf(0, 100, 0, 0, 100, annual_production=100)
    assert res["scrapped_mass_annual_kg"] == 20.0
    assert res["ghg_rework"] == 80.0  # only the 80 parts that were not scrapped are reworked


def test_prior_reweighting_uses_rates_not_test_prevalence():
    from src.sustainability.qcf_engine import QualityCarbonFootprintEngine, SustainabilityParameters

    p = SustainabilityParameters(defect_prior=0.01, P_edge=0.0)
    eng = QualityCarbonFootprintEngine(params=p)
    # a defect-heavy test set (70% defective) with TPR 0.9 and FPR 0.1
    res = eng.compute_annual_qcf(tp_count=63, fp_count=3, fn_count=7, tn_count=27, total_parts=100,
                                 annual_production=1_000_000)
    expected_fn = 0.1 * 0.01 * 1_000_000
    per_escape = p.theta_tier * p.m_part * p.kappa_mat + p.escape_rework_factor * p.E_rework * p.xi_grid
    assert abs(res["ghg_escape"] - expected_fn * per_escape) < 1e-6


def test_compute_footprint_has_no_side_effects():
    from src.sustainability.qcf_engine import QualityCarbonFootprintEngine

    eng = QualityCarbonFootprintEngine()
    before = eng.params.latency_ms
    eng.compute_footprint(tp=5, tn=90, fp=3, fn=2, latency_sec=0.5)
    assert eng.params.latency_ms == before
