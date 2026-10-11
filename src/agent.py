"""
agent.py —— OpenClaw 智能体层：把诊断报文"翻译"成现场话术

严格对应《诊断接口对接说明 v1.0》的三条硬约束：
    ① 不做诊断 —— 故障类型/健康度/风险等级只取自诊断报文，不自行推断；
    ② 不越权执行 —— 停机、切泵只"建议并等待人工确认"；
    ③ 不承诺无法验证的数字 —— 无模型输出/无实测支撑的节能率、寿命不写入。

本文件是**离线可复现的规则版话术生成器**，用于演示智能体层行为；
真实部署时这一层由 OpenClaw（本地小模型 + 机理知识库）承担，
本模块可作为其系统提示词的"最小可用参考实现"。
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    from .health import FAULT_CN, GRADE, grade_of
except ImportError:
    from health import FAULT_CN, GRADE, grade_of


# 故障 → 现场可执行建议（机理知识库最小集）
ACTION_DB = {
    "bearing_outer": [
        "记录驱动端轴承温度与振动趋势，加密巡检至每日 2 次",
        "准备同型号轴承备件，安排下次计划检修更换",
    ],
    "bearing_inner": [
        "确认润滑脂状态（是否缺脂/变质）",
        "准备轴承备件，安排计划检修更换",
    ],
    "unbalance": [
        "核对叶轮/转子是否有结垢、磨损或异物附着",
        "评估动平衡复校需求（建议计划停机后做）",
    ],
    "misalignment": [
        "安排激光对中复测（联轴器）",
        "检查地脚螺栓与垫片是否松动",
    ],
    "looseness": [
        "逐条紧固地脚螺栓与连接法兰",
        "检查基础/垫铁是否有松动或裂纹",
    ],
    "cavitation": [
        "检查入口压力与有效汽蚀余量（NPSHa）是否满足",
        "排查入口滤网堵塞、液位过低或介质温度偏高",
    ],
    "seal_wear": [
        "检查柱塞密封处泄漏量",
        "准备密封件备件，安排更换",
    ],
    "normal": ["维持当前运行与巡检节奏"],
}


def _risk_badge(risk: str) -> str:
    return {
        "normal": "🟢 正常", "low": "🟡 轻微", "medium": "🟠 中等",
        "high": "🔴 严重", "critical": "⛔ 危急",
    }.get(risk, risk)


def generate_script(report: dict, device_name: str | None = None,
                    rated_rpm: float | None = None) -> str:
    """
    诊断报文 → 四段式现场话术（结论 / 依据 / 建议 / 风险）。
    严格只读报文，不补充报文之外的数字。
    """
    dev = device_name or report.get("device_id", "设备")
    score = report["health_score"]
    risk = report["risk_level"]
    faults = report.get("fault_type", ["normal"])
    conf = report.get("confidence", 0.0)
    ev = report.get("key_evidence", [])
    life = report.get("remaining_life_days")

    fault_txt = "、".join(FAULT_CN.get(f, f) for f in faults)
    risk_txt, sem, act_level = grade_of(score)

    lines: list[str] = []
    if report.get("degraded"):
        lines.append("⚠️ 数据降级：本次结论基于降级模型/不完整数据，仅供参考。")
    lines.append(f"【{dev}】{_risk_badge(risk)}　健康度 {score}/100（{sem}）")

    # --- 结论 ---
    if faults == ["normal"]:
        lines.append(f"① 结论：设备状态正常。（置信度 {conf:.0%}）")
    else:
        lines.append(f"① 结论：检出异常 —— {fault_txt}。（置信度 {conf:.0%}）")

    # --- 依据 ---
    if ev:
        parts = []
        for e in ev[:4]:
            unit = e.get("unit", "")
            base = e.get("baseline")
            if base is not None and e.get("deviation") is not None:
                sign = "↑" if e["deviation"] > 0 else "↓"
                parts.append(f"{e['metric']}={e['value']}{unit}（基线 {base}，{sign}{abs(e['deviation'])*100:.0f}%）")
            else:
                parts.append(f"{e['metric']}={e['value']}{unit}")
        lines.append("② 依据：" + "；".join(parts))
    else:
        lines.append("② 依据：（无有效证据，不下结论）")

    # --- 建议（只建议，不下令） ---
    acts = ACTION_DB.get(faults[0], ACTION_DB["normal"])
    lines.append("③ 建议：" + "；".join(acts) + "。")

    # --- 风险与权限边界 ---
    if life and risk in ("high", "critical"):
        lines.append(f"④ 风险：预计剩余寿命约 {life[0]}~{life[1]} 天，动作等级 {act_level}。")
    else:
        lines.append(f"④ 风险：动作等级 {act_level}。")

    if risk in ("high", "critical"):
        lines.append("⚠️ 停机/切泵属高权限动作，本系统仅建议，须经人工确认后执行。")
    else:
        lines.append("（本系统不下发任何控制指令；保护动作由设备既有安全回路负责。）")

    return "\n".join(lines)


if __name__ == "__main__":
    import json
    from diagnose import diagnose_state  # type: ignore
    for s in ["normal", "bearing_outer", "unbalance", "cavitation"]:
        rep = diagnose_state(s, device_id="PMP-01 注水A泵")
        print("=" * 72)
        print(generate_script(rep))
