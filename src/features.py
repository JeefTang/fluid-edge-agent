"""
features.py —— 时域 + 频域 + 包络谱特征提取

设计原则：
1. **只用 numpy**（不依赖 scipy/sklearn），保证在 RK3576 边缘盒子上也能跑；
2. 特征输入维度小（<20），便于后续 INT8 量化上 NPU；
3. 全部物理可解释——每个特征对应一类失效机理。

Hilbert 变换用 FFT 手写实现（numpy 版解析信号），避免引入 scipy。
"""

from __future__ import annotations

import os
import sys

import numpy as np

# 允许 `python3 src/features.py` 直接运行，也允许作为包导入
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    from .signal_sim import Bearing
except ImportError:  # 直接运行脚本时
    from signal_sim import Bearing


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def hilbert(x: np.ndarray) -> np.ndarray:
    """纯 numpy 实现解析信号（Hilbert 变换）。"""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    spectra = np.fft.fft(x)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1
        h[1:n // 2] = 2
    else:
        h[0] = 1
        h[1:(n + 1) // 2] = 2
    return np.fft.ifft(spectra * h)


def envelope_spectrum(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """包络谱（Hilbert 包络 → 去直流 → FFT）。用于捕捉轴承冲击频率。"""
    env = np.abs(hilbert(x))
    env = env - env.mean()
    spec = np.abs(np.fft.rfft(env))
    freqs = np.fft.rfftfreq(len(x), 1 / fs)
    return freqs, spec


def amplitude_spectrum(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """常规幅值谱。"""
    spec = np.abs(np.fft.rfft(x)) * 2 / len(x)
    freqs = np.fft.rfftfreq(len(x), 1 / fs)
    return freqs, spec


def _peak_near(freqs: np.ndarray, spec: np.ndarray, target: float,
               tol: float = 2.0) -> float:
    """取 target 频率 ±tol 范围内的谱峰幅值。"""
    m = (freqs >= target - tol) & (freqs <= target + tol)
    if not np.any(m):
        return 0.0
    return float(spec[m].max())


# ---------------------------------------------------------------------------
# 时域特征
# ---------------------------------------------------------------------------

def time_features(x: np.ndarray) -> dict:
    """时域统计特征（对冲击型故障敏感）。"""
    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.max(np.abs(x)))
    mean_abs = float(np.mean(np.abs(x)))
    crest = peak / rms if rms > 1e-12 else 0.0          # 峰值因子
    # 峭度（正态分布 ≈ 3，冲击型故障显著升高）
    m4 = float(np.mean((x - x.mean()) ** 4))
    m2 = float(np.mean((x - x.mean()) ** 2))
    kurtosis = m4 / (m2 ** 2) if m2 > 1e-18 else 3.0
    skewness = float(np.mean((x - x.mean()) ** 3) / (m2 ** 1.5)) if m2 > 1e-18 else 0.0
    return {
        "rms": rms,
        "peak": peak,
        "mean_abs": mean_abs,
        "crest": crest,
        "kurtosis": kurtosis,
        "skewness": skewness,
        "shape": mean_abs / rms if rms > 1e-12 else 0.0,
    }


# ---------------------------------------------------------------------------
# 频域 / 包络谱特征
# ---------------------------------------------------------------------------

def frequency_features(x: np.ndarray, fs: float, rpm: float) -> dict:
    """
    频域特征。以 1x 为基准，输出各阶次相对幅值 + 轴承特征频率包络能量。
    """
    fr = rpm / 60.0
    freqs, amp = amplitude_spectrum(x, fs)
    env_freqs, env_spec = envelope_spectrum(x, fs)

    # 转频阶次（相对 1x 归一化）
    a1x = _peak_near(freqs, amp, fr)
    a2x = _peak_near(freqs, amp, 2 * fr)
    a3x = _peak_near(freqs, amp, 3 * fr)
    a_half = _peak_near(freqs, amp, 0.5 * fr)

    # 轴承特征频率（包络谱）
    bcf = Bearing().characteristic_frequencies(rpm)
    bpfo = _peak_near(env_freqs, env_spec, bcf["BPFO"])
    bpfi = _peak_near(env_freqs, env_spec, bcf["BPFI"])
    bsf = _peak_near(env_freqs, env_spec, bcf["BSF"])

    # ★ 包络谱峰"突出度" = 特征频率峰 / 包络谱中位数
    # 轴承故障：包络窄带 → 峰极突出（>12）；汽蚀等宽带：峰不突出（<5）
    env_med = float(np.median(env_spec)) or 1e-9
    bpfo_prom = bpfo / env_med
    bpfi_prom = bpfi / env_med

    # 高频宽带能量占比（汽蚀指标）：2~8 kHz 能量 / 总能量
    hi = (freqs >= 2000) & (freqs <= 8000)
    hi_ratio = float(np.sum(amp[hi] ** 2) / (np.sum(amp ** 2) + 1e-12))

    return {
        "a_1x": a1x, "a_2x": a2x, "a_3x": a3x, "a_halfx": a_half,
        "r_2x_1x": a2x / a1x if a1x > 1e-9 else 0.0,      # 不对中指标
        "r_halfx_1x": a_half / a1x if a1x > 1e-9 else 0.0,  # 松动指标
        "bpfo_env": bpfo, "bpfi_env": bpfi, "bsf_env": bsf,
        "bpfo_prom": bpfo_prom, "bpfi_prom": bpfi_prom,
        "hi_band_ratio": hi_ratio,
        "fr": fr, "BPFO": bcf["BPFO"], "BPFI": bcf["BPFI"],
    }


def extract(x: np.ndarray, fs: float, rpm: float) -> dict:
    """提取完整特征向量（时域 + 频域），返回扁平 dict。"""
    feat = {}
    feat.update(time_features(x))
    feat.update(frequency_features(x, fs, rpm))
    return feat


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.signal_sim import generate, STATES, STATE_CN
    for s in STATES:
        x = generate(s)
        f = extract(x, 25_600, 1480.0)
        print(f"{STATE_CN[s]:8s} rms={f['rms']:.3f} kurt={f['kurtosis']:.2f} "
              f"crest={f['crest']:.2f} bpfo={f['bpfo_env']:.1f} "
              f"r2x1x={f['r_2x_1x']:.2f} hi={f['hi_band_ratio']:.3f}")
