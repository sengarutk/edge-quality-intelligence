from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class ThresholdRecord:
    threshold_type: str  # "quantile_99", "oracle_f1", "cost_optimal_val", "alert_budget_5"
    source_split: str    # "train_normal", "val_nominal", "val_split", "test_oracle"
    uses_test_labels: bool
    category: str
    method: str
    seed: int
    threshold_value: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_deployable(self) -> bool:
        """Returns True if the threshold can be deployed in production without test-label leakage."""
        return not self.uses_test_labels and self.source_split in ["train_normal", "val_nominal", "val_split"]


class ThresholdLineageAuditor:
    def __init__(self):
        self.records: List[ThresholdRecord] = []

    def record(self, rec: ThresholdRecord):
        self.records.append(rec)

    def audit_leakage(self) -> List[ThresholdRecord]:
        """Returns any records targeted for deployable splits that illegally use test labels."""
        violations = [
            r for r in self.records 
            if r.uses_test_labels and r.source_split in ["train_normal", "val_nominal", "val_split"]
        ]
        return violations

    def summary(self) -> Dict[str, int]:
        total = len(self.records)
        deployable = sum(1 for r in self.records if r.is_deployable())
        leaked = len(self.audit_leakage())
        return {
            "total_thresholds": total,
            "deployable_thresholds": deployable,
            "leakage_violations": leaked
        }