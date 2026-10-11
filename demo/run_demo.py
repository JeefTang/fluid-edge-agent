"""
run_demo.py —— 一键演示：仿真信号 → 机理诊断 → 契约报文 → 现场话术 → 图表

用法：
    python3 demo/run_demo.py              # 完整演示，输出图表到 demo/out/
    python3 demo/run_demo.py --no-plot    # 只打文本

产出：
    demo/out/demo_chart.png               四联图（时域对比 / 包络谱 / 健康度）
    examples/diagnosis_sample.json        示例诊断报文（契约格式）
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import features as ft          # noqa: E402
from src import signal_sim as sim       # noqa: E402
from src.agent import generate_script   # noqa: E402
from src.diagnose import diagnose_signal  # noqa: E402
from src.health import FAULT_CN, Diagnoser, grade_of  # noqa: E402

RPM = 1480.0


def _setup_cjk_font() -> None:
    """配置 matplotlib 中文字体（自适应常见发行版）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    prefer = ["Noto Sans CJK SC", "Noto Serif CJK SC", "WenQuanYi Zen Hei",
              "WenQuanYi Micro Hei", "Source Han Sans SC", "SimHei"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in prefer:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def run_all(rpm: float = RPM) -> list[dict]:
    """遍历全部工况，返回诊断结果列表。"""
    diag = Diagnoser(rpm=rpm)
    results = []
    for s in sim.STATES:
        x = sim.generate(s, rpm=rpm)
        feat = ft.extract(x, sim.FS, rpm)
        res = diag.diagnose(feat)
        report = diagnose_signal(x, rpm=rpm, device_id="PMP-01 注水A泵",
                                 diagnoser=diag)
        risk, sem, act = grade_of(res["health_score"])
        results.append({
            "state": s, "signal": x, "feat": feat, "result": res,
            "report": report, "risk": risk, "sem": sem, "act": act,
            "hit": (s in res["fault_type"]) or (s == "normal" and res["fault_type"] == ["normal"]),
        })
    return results


def print_text(results: list[dict]) -> None:
    print("=" * 88)
    print("流体设备边缘智能体 · 机理诊断 PoC —— 演示")
    print("=" * 88)
    print(f"{'仿真工况':16s} {'诊断结论':28s} {'健康度':>5s} {'风险':>8s} {'置信':>5s} {'命中'}")
    print("-" * 88)
    hit = 0
    for r in results:
        faults = "/".join(FAULT_CN.get(f, f) for f in r["result"]["fault_type"])
        hit += r["hit"]
        print(f"{sim.STATE_CN[r['state']]:16s} {faults:28s} "
              f"{r['result']['health_score']:5d} {r['risk']:>8s} "
              f"{r['result']['confidence']:5.2f} {'✅' if r['hit'] else '❌'}")
    print("-" * 88)
    print(f"命中率：{hit}/{len(results)}")
    print()

    # 展示一条完整话术
    demo = next(r for r in results if r["state"] == "bearing_outer")
    print("【智能体层输出示例 —— 轴承外圈缺陷】")
    print(generate_script(demo["report"]))
    print()


def make_chart(results: list[dict], out_path: str) -> None:
    _setup_cjk_font()
    import matplotlib.pyplot as plt

    from src.signal_sim import Bearing
    bcf = Bearing().characteristic_frequencies(RPM)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    t_ms = np.arange(int(0.05 * sim.FS)) / sim.FS * 1000

    # (a) 正常 时域
    ax = axes[0, 0]
    xn = next(r for r in results if r["state"] == "normal")["signal"]
    ax.plot(t_ms, xn[:len(t_ms)], lw=0.7, color="#2b8a3e")
    ax.set_title("正常工况 · 振动时域（50 ms）")
    ax.set_xlabel("时间 (ms)"); ax.set_ylabel("加速度 (g)")
    ax.grid(alpha=0.3)

    # (b) 轴承外圈 时域（冲击可见）
    ax = axes[0, 1]
    xo = next(r for r in results if r["state"] == "bearing_outer")["signal"]
    ax.plot(t_ms, xo[:len(t_ms)], lw=0.7, color="#c92a2a")
    ax.set_title("轴承外圈缺陷 · 振动时域（周期性冲击）")
    ax.set_xlabel("时间 (ms)"); ax.set_ylabel("加速度 (g)")
    ax.grid(alpha=0.3)

    # (c) 包络谱对比
    ax = axes[1, 0]
    for st, color in [("normal", "#2b8a3e"), ("bearing_outer", "#c92a2a"),
                      ("bearing_inner", "#e8590c")]:
        x = next(r for r in results if r["state"] == st)["signal"]
        ef, es = ft.envelope_spectrum(x, sim.FS)
        m = ef <= 300
        ax.plot(ef[m], es[m], lw=0.9, color=color, label=sim.STATE_CN[st])
    for f, name, c in [(bcf["BPFO"], "BPFO", "#c92a2a"), (bcf["BPFI"], "BPFI", "#e8590c")]:
        ax.axvline(f, ls="--", lw=0.8, color=c, alpha=0.7)
        ax.text(f, ax.get_ylim()[1] * 0.9, name, color=c, fontsize=9)
    ax.set_title("包络谱对比 —— 轴承故障呈窄带峰")
    ax.set_xlabel("频率 (Hz)"); ax.set_ylabel("包络幅值")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (d) 健康度柱状图
    ax = axes[1, 1]
    labels = [sim.STATE_CN[r["state"]] for r in results]
    scores = [r["result"]["health_score"] for r in results]
    colors = ["#2b8a3e" if s >= 90 else "#f08c00" if s >= 60 else "#c92a2a" for s in scores]
    bars = ax.barh(labels, scores, color=colors)
    for b, s in zip(bars, scores):
        ax.text(b.get_width() + 1, b.get_y() + b.get_height() / 2, str(s),
                va="center", fontsize=9)
    ax.set_xlim(0, 115)
    ax.set_title("健康度评分（机理规则诊断）")
    ax.set_xlabel("health_score（0~100）")
    ax.grid(axis="x", alpha=0.3)

    fig.suptitle("流体设备边缘智能体 PoC —— 机理仿真 → 特征 → 规则诊断", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130)
    print(f"图表已输出：{out_path}")


def dump_sample(results: list[dict], out_path: str) -> None:
    sample = next(r for r in results if r["state"] == "bearing_outer")["report"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)
    print(f"示例报文已输出：{out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="流体设备边缘智能体 PoC 演示")
    ap.add_argument("--no-plot", action="store_true", help="跳过图表生成")
    ap.add_argument("--out", default=os.path.join(ROOT, "demo", "out", "demo_chart.png"))
    args = ap.parse_args()

    results = run_all()
    print_text(results)
    dump_sample(results, os.path.join(ROOT, "examples", "diagnosis_sample.json"))
    if not args.no_plot:
        try:
            make_chart(results, args.out)
        except Exception as e:  # 出图失败不应影响主流程
            print(f"[warn] 图表生成失败：{e}")


if __name__ == "__main__":
    main()
