"""
health.py —— 健康度评分 + 机理规则诊断

本模块实现**可解释的机理规则诊断**（不是黑箱神经网络），对应项目的
"设备机理"壁垒：

    特征 → 规则 → 故障类型 / 健康度 / 置信度 / 风险等级 / 证据

为什么先用规则而不是深度学习？
    现场真实失效样本稀缺（复现故障要停机），深度模型冷启动没有数据。
    先用机理规则 + 无监督异常检测顶上，随样本积累再迭代模型。

核心判别器说明
--------------
* **包络谱峰突出度** bpfo_prom / bpfi_prom（= 特征频率峰 / 包络谱中位数）
  —— 轴承缺陷会产生**窄带**包络 → 峰极突出（实测 >12）；汽蚀等**宽带**
  现象不产生特定窄带峰 → 突出度低（实测 <5）。这是区分"轴承 vs 汽蚀"的
  关键（两者都会抬高高频能量，只看总能量会混淆）。
* **转频阶次比** r_2x_1x（不对中）、r_halfx_1x（机械松动）
* **高频带能量比** hi_band_ratio（2~8 kHz，汽蚀）

⚠️ 所有阈值均为**演示值**，真实项目必须用现场实测基线替换（标定）。
"""

from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    from . import features as ft
    from .signal_sim import FS
except ImportError:
    import features as ft
    from signal_sim import FS


# ---------------------------------------------------------------------------
# 冻结口径（与《诊断接口对接说明 v1.0》一致）
# ---------------------------------------------------------------------------

GRADE = [
    (90, "normal",   "正常",        "L0 仅记录"),
    (75, "low",      "轻微偏离基线", "L1 记录并观察"),
    (60, "medium",   "明确劣化",    "L2 自动工单"),
    (40, "high",     "严重风险",    "L3 建议降负荷＋告警"),
    (0,  "critical", "临近失效",    "L4 建议停机（封锁启机）"),
]

FAULT_CN = {
    "normal": "正常", "bearing_outer": "轴承外圈缺陷", "bearing_inner": "轴承内圈缺陷",
    "seal_wear": "柱塞密封磨损/泄漏", "valve_wear": "阀组磨损/敲击异常",
    "cavitation": "汽蚀", "misalignment": "联轴器不对中", "unbalance": "转子不平衡",
    "looseness": "机械松动", "overload": "过载", "clog": "堵塞/卡阀",
    "electrical_fault": "电机/变频器侧异常", "sensor_drift": "传感器漂移",
    "unclassified": "未分类异常",
}

HINT = {
    "bearing_outer": "check_drive_end_bearing", "bearing_inner": "check_drive_end_bearing",
    "unbalance": "check_rotor_balance", "misalignment": "check_coupling_alignment",
    "looseness": "check_mounting_bolts", "cavitation": "check_inlet_pressure_npsh",
    "seal_wear": "check_seal_leakage",
}


def grade_of(health_score: float) -> tuple[str, str, str]:
    """按健康度返回 (risk_level, 语义, 默认动作等级)。"""
    for lo, risk, sem, act in GRADE:
        if health_score >= lo:
            return risk, sem, act
    return "critical", "临近失效", "L4 建议停机（封锁启机）"


# ---------------------------------------------------------------------------
# 基线
# ---------------------------------------------------------------------------

def baseline_features(rpm: float = 1480.0, fs: int = FS) -> dict:
    """参考基线（健康态特征）。真实项目中替换为**现场健康工况实测**。"""
    try:
        from .signal_sim import generate  # type: ignore
    except ImportError:
        from signal_sim import generate  # type: ignore
    x = generate("normal", rpm=rpm, fs=fs, seed=7)
    return ft.extract(x, fs, rpm)


# ---------------------------------------------------------------------------
# 规则诊断
# ---------------------------------------------------------------------------

# 演示阈值（真实项目须标定）
TH = {
    "bpfo_prom": 12.0,       # 包络峰突出度：轴承外圈
    "bpfi_prom": 12.0,       # 包络峰突出度：轴承内圈
    "halfx": 0.12,           # 0.5x/1x 比：机械松动
    "r2x1x": 0.50,           # 2x/1x 比：不对中
    "hi_band": 0.35,         # 高频带能量比：汽蚀
    "rms_r": 1.80,           # 相对基线 RMS 倍率：不平衡
}


class Diagnoser:
    """机理规则诊断器。"""

    def __init__(self, rpm: float = 1480.0, fs: int = FS, baseline: dict | None = None):
        self.rpm = rpm
        self.fs = fs
        self.baseline = baseline or baseline_features(rpm, fs)

    @staticmethod
    def _ev(metric, value, baseline, unit, window="1s窗", dev=None) -> dict:
        if dev is None and baseline not in (0, None):
            dev = round((value - baseline) / abs(baseline), 3)
        return {"metric": metric, "value": round(float(value), 3),
                "baseline": round(float(baseline), 3) if baseline is not None else None,
                "deviation": dev, "window": window, "unit": unit}

    def diagnose(self, feat: dict) -> dict:
        b = self.baseline
        ev: list[dict] = []
        faults: list[tuple[str, int]] = []   # (fault, 扣分)

        bpfo_prom = feat.get("bpfo_prom", 0.0)
        bpfi_prom = feat.get("bpfi_prom", 0.0)
        r_halfx = feat["r_halfx_1x"]
        r2x1x = feat["r_2x_1x"]
        hi = feat["hi_band_ratio"]
        rms_r = feat["rms"] / max(b["rms"], 1e-9)
        a1x_r = feat["a_1x"] / max(b["a_1x"], 1e-9)

        # --- 1. 机械松动（0.5x 与高次谐波族）优先级最高 ---
        if r_halfx > TH["halfx"]:
            d = int(20 + 25 * np.clip((r_halfx - TH["halfx"]) / 0.8, 0, 1))
            faults.append(("looseness", d))
            ev.append(self._ev("r_halfx_1x", r_halfx, b["r_halfx_1x"], "-"))

        # --- 2. 轴承缺陷（窄带包络峰突出）---
        elif max(bpfo_prom, bpfi_prom) > TH["bpfo_prom"]:
            if bpfo_prom >= bpfi_prom:
                fault, prom, base = "bearing_outer", bpfo_prom, b["bpfo_env"]
                ev.append(self._ev("bpfo_prom", bpfo_prom, b.get("bpfo_prom"), "×中位数"))
            else:
                fault, prom, base = "bearing_inner", bpfi_prom, b["bpfi_env"]
                ev.append(self._ev("bpfi_prom", bpfi_prom, b.get("bpfi_prom"), "×中位数"))
            d = int(30 + 25 * np.clip((prom - 12) / 45, 0, 1))
            faults.append((fault, d))
            ev.append(self._ev("kurtosis", feat["kurtosis"], b["kurtosis"], "-"))

        # --- 3. 汽蚀（高频宽带抬升、无窄带峰）---
        elif hi > TH["hi_band"]:
            d = int(25 + 25 * np.clip((hi - TH["hi_band"]) / 0.5, 0, 1))
            faults.append(("cavitation", d))
            ev.append(self._ev("hi_band_ratio", hi, b["hi_band_ratio"], "-"))

        # --- 4. 转子不平衡（整体抬升但 1x 主导）---
        elif rms_r > TH["rms_r"] and a1x_r > 1.8 and r2x1x < 0.35:
            d = int(25 + 25 * np.clip((rms_r - 1.8) / 2.5, 0, 1))
            faults.append(("unbalance", d))
            ev.append(self._ev("a_1x", feat["a_1x"], b["a_1x"], "g"))
            ev.append(self._ev("rms", feat["rms"], b["rms"], "g"))

        # --- 5. 联轴器不对中（2x 抬升）---
        elif r2x1x > TH["r2x1x"]:
            d = int(20 + 25 * np.clip((r2x1x - TH["r2x1x"]) / 1.0, 0, 1))
            faults.append(("misalignment", d))
            ev.append(self._ev("r_2x_1x", r2x1x, b["r_2x_1x"], "-"))

        # --- 汇总 ---------------------------------------------------------
        if not faults:
            return {
                "health_score": int(np.clip(round(100 - 3 * abs(rms_r - 1)), 90, 100)),
                "fault_type": ["normal"], "confidence": 0.95,
                "key_evidence": [self._ev("rms", feat["rms"], b["rms"], "g"),
                                 self._ev("kurtosis", feat["kurtosis"], b["kurtosis"], "-")],
                "suggestion_hint": None, "severity_map": {},
            }

        health = int(np.clip(100 - sum(d for _, d in faults), 5, 100))
        faults.sort(key=lambda t: -t[1])
        fault_list = [f for f, _ in faults]

        if health >= 75:
            life = [90, 180]
        elif health >= 60:
            life = [30, 90]
        elif health >= 40:
            life = [7, 30]
        else:
            life = [1, 7]

        return {
            "health_score": health,
            "fault_type": fault_list,
            "confidence": round(float(min(0.97, 0.6 + 0.35 * min(sum(d for _, d in faults), 60) / 60)), 2),
            "key_evidence": ev[:6],
            "suggestion_hint": HINT.get(fault_list[0]),
            "remaining_life_days": life,
            "severity_map": {f: d for f, d in faults},
        }


if __name__ == "__main__":
    try:
        from .signal_sim import generate, STATES  # type: ignore
    except ImportError:
        from signal_sim import generate, STATES  # type: ignore
    d = Diagnoser()
    for s in STATES:
        x = generate(s)
        f = ft.extract(x, FS, 1480.0)
        r = d.diagnose(f)
        risk, sem, act = grade_of(r["health_score"])
        print(f"{s:16s} → {str(r['fault_type']):24s} score={r['health_score']:3d} "
              f"risk={risk:8s} conf={r['confidence']:.2f}")
