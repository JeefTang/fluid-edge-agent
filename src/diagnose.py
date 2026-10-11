"""
diagnose.py —— 诊断主入口：信号 → 契约化 JSON

把「采集 → 特征 → 机理诊断」串起来，输出**对齐《诊断接口对接说明 v1.0》**
的冻结字段报文。这就是智能体层的唯一输入。

报文示例见 examples/diagnosis_sample.json。
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone, timedelta

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    from . import features as ft
    from . import signal_sim as sim
    from .health import Diagnoser, grade_of
except ImportError:
    import features as ft
    import signal_sim as sim
    from health import Diagnoser, grade_of

SCHEMA_VERSION = "1.0"
MODEL_VERSION = "fluid_diag_rulebased_v0.1"
CN_TZ = timezone(timedelta(hours=8))


def _now_iso() -> str:
    return datetime.now(CN_TZ).replace(microsecond=0).isoformat()


def build_report(feat: dict, result: dict, device_id: str,
                 ts: str | None = None, degraded: bool = False) -> dict:
    """按冻结契约组装诊断报文。"""
    risk, _sem, _act = grade_of(result["health_score"])
    report = {
        "schema_version": SCHEMA_VERSION,
        "ts": ts or _now_iso(),
        "device_id": device_id,
        "health_score": int(result["health_score"]),
        "fault_type": list(result["fault_type"]),
        "confidence": float(result["confidence"]),
        "risk_level": risk,
        "remaining_life_days": result.get("remaining_life_days"),
        "key_evidence": result.get("key_evidence", [])[:6],
        "suggestion_hint": result.get("suggestion_hint"),
        "model_version": MODEL_VERSION,
        "degraded": degraded,
    }
    return report


def diagnose_signal(x: np.ndarray, rpm: float = 1480.0, device_id: str = "PMP-01",
                    fs: int = sim.FS, diagnoser: Diagnoser | None = None,
                    ts: str | None = None, degraded: bool = False) -> dict:
    """原始振动信号 → 契约化诊断报文。"""
    d = diagnoser or Diagnoser(rpm=rpm, fs=fs)
    feat = ft.extract(x, fs, rpm)
    res = d.diagnose(feat)
    return build_report(feat, res, device_id, ts=ts, degraded=degraded)


def diagnose_state(state: str, rpm: float = 1480.0, device_id: str = "PMP-01",
                   seed: int | None = 42, diagnoser: Diagnoser | None = None) -> dict:
    """便捷函数：按工况标签仿真一段信号并诊断（演示用）。"""
    x = sim.generate(state, rpm=rpm, seed=seed)
    return diagnose_signal(x, rpm=rpm, device_id=device_id, diagnoser=diagnoser)


if __name__ == "__main__":
    from health import FAULT_CN  # type: ignore
    for s in sim.STATES:
        rep = diagnose_state(s, device_id=f"PMP-{s}")
        faults = "/".join(FAULT_CN.get(f, f) for f in rep["fault_type"])
        print(f"{s:16s} → {faults:16s} score={rep['health_score']:3d} "
              f"risk={rep['risk_level']:9s} conf={rep['confidence']:.2f} "
              f"hint={rep['suggestion_hint']}")
