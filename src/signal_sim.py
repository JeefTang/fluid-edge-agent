"""
signal_sim.py —— 旋转机械振动信号机理仿真

本模块是 PoC 的"数据来源"。真实项目中数据来自现场加速度计（IEPE）；
为了让仓库**零数据依赖、开箱即跑**，这里用滚动轴承运动学 + 转子动力学
生成 6 类典型工况的仿真振动信号。

这正是本项目宣称的壁垒之一：设备机理（而不是通用平台）。

轴承特征频率（基于滚动轴承几何，SKF 6205 典型参数）：
    BPFO (外圈通过频率) = (N/2) * fr * (1 - (d/D) * cos(alpha))
    BPFI (内圈通过频率) = (N/2) * fr * (1 + (d/D) * cos(alpha))
    BSF  (滚动体自转频率) = (D / (2d)) * fr * (1 - ((d/D) * cos(alpha))**2)
    FTF  (保持架频率)   = (fr / 2) * (1 - (d/D) * cos(alpha))

其中 fr = 转轴转频 (Hz)，N = 滚珠数，d = 滚珠直径，D = 节圆直径。

⚠️ 边界声明：仿真信号用于演示算法链路，**不能替代现场标定**。
   所有阈值在真实项目中必须用现场实测基线替换。
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# 默认物理参数
# ---------------------------------------------------------------------------

FS = 25_600            # 采样率 Hz（工业振动常规 >= 10 kHz，这里取 25.6 kHz）
DURATION = 1.0         # 采样时长 s


class Bearing:
    """深沟球轴承几何参数（默认取 SKF 6205 典型值）。"""

    def __init__(self, n_balls: int = 9, d: float = 7.94, D: float = 39.04,
                 contact_angle_deg: float = 0.0):
        self.n_balls = n_balls          # 滚珠数 N
        self.d = d                      # 滚珠直径 mm
        self.D = D                      # 节圆直径 mm
        self.alpha = np.deg2rad(contact_angle_deg)

    def characteristic_frequencies(self, rpm: float) -> dict:
        """给定转速 (rpm) 返回各特征频率 (Hz)。"""
        fr = rpm / 60.0
        ratio = self.d / self.D * np.cos(self.alpha)
        return {
            "fr": fr,
            "BPFO": (self.n_balls / 2) * fr * (1 - ratio),
            "BPFI": (self.n_balls / 2) * fr * (1 + ratio),
            "BSF": (self.D / (2 * self.d)) * fr * (1 - ratio ** 2),
            "FTF": (fr / 2) * (1 - ratio),
        }


# ---------------------------------------------------------------------------
# 信号生成
# ---------------------------------------------------------------------------

def _impulse_train(fs: int, n: int, freq: float, decay: float,
                   phase: float = 0.0) -> np.ndarray:
    """生成周期性冲击串：每次冲击为指数衰减振荡，用于轴承缺陷建模。"""
    t = np.arange(n) / fs
    train = np.zeros(n)
    period = fs / freq
    # 每次冲击的起始样点
    starts = np.arange(0, n, period)
    for s in starts:
        s = int(round(s + phase * period))
        if s >= n:
            continue
        idx = np.arange(n - s)
        wave = np.exp(-decay * idx / fs) * np.sin(2 * np.pi * fs / 8 * idx / fs)
        train[s:] += wave[:n - s]
    return train


def generate(state: str = "normal", rpm: float = 1480.0, fs: int = FS,
             duration: float = DURATION, seed: int | None = 42,
             noise: float = 0.05) -> np.ndarray:
    """
    生成指定工况的振动加速度信号。

    Parameters
    ----------
    state : str
        工况标签，取值：
        normal / bearing_outer / bearing_inner / unbalance /
        misalignment / looseness / cavitation
    rpm : float
        转轴转速 (rpm)
    fs : int
        采样率 (Hz)
    duration : float
        时长 (s)
    seed : int | None
        随机种子，保证可复现
    noise : float
        宽带噪声水平（相对幅值）

    Returns
    -------
    np.ndarray
        振动加速度信号（归一化单位）
    """
    rng = np.random.default_rng(seed)
    n = int(fs * duration)
    t = np.arange(n) / fs

    bearing = Bearing()
    f = bearing.characteristic_frequencies(rpm)
    fr = f["fr"]

    sig = np.zeros(n)

    # --- 基础旋转分量（所有工况都有 1x） ---
    sig += 0.30 * np.sin(2 * np.pi * fr * t)

    if state == "normal":
        # 健康：仅 1x 为主 + 极少量 2x
        sig += 0.03 * np.sin(2 * np.pi * 2 * fr * t)

    elif state == "bearing_outer":
        # 外圈缺陷：BPFO 冲击串，幅值稳定（不受轴转调制）
        sig += 0.55 * _impulse_train(fs, n, f["BPFO"], decay=900)

    elif state == "bearing_inner":
        # 内圈缺陷：BPFI 冲击串，被 1x 调制（内圈随轴旋转进出承载区）
        train = _impulse_train(fs, n, f["BPFI"], decay=900)
        sig += 0.50 * train * (1 + 0.8 * np.sin(2 * np.pi * fr * t))

    elif state == "unbalance":
        # 转子不平衡：1x 幅值显著升高，2x 略有
        sig += 1.10 * np.sin(2 * np.pi * fr * t)
        sig += 0.10 * np.sin(2 * np.pi * 2 * fr * t + 0.5)

    elif state == "misalignment":
        # 联轴器不对中：2x 抬升，常伴 1x
        sig += 0.45 * np.sin(2 * np.pi * 2 * fr * t + 0.3)
        sig += 0.12 * np.sin(2 * np.pi * fr * t)

    elif state == "looseness":
        # 机械松动：出现 0.5x 与高次谐波族
        sig += 0.35 * np.sin(2 * np.pi * 0.5 * fr * t)
        sig += 0.30 * np.sin(2 * np.pi * 3 * fr * t + 1.1)
        sig += 0.25 * np.sin(2 * np.pi * 4 * fr * t + 0.7)
        sig += 0.012 * np.sign(np.sin(2 * np.pi * fr * t))  # 削顶冲击

    elif state == "cavitation":
        # 汽蚀：2~8 kHz 宽带高频能量抬升 + 随机冲击
        band = rng.standard_normal(n)
        # 简易带通：高频段保留
        from numpy.fft import rfft, irfft, rfftfreq
        spec = rfft(band)
        freqs = rfftfreq(n, 1 / fs)
        spec[(freqs < 2000) | (freqs > 8000)] = 0
        sig += 0.55 * irfft(spec, n)
    else:
        raise ValueError(f"未知工况: {state}")

    # --- 宽带噪声 ---
    sig += noise * rng.standard_normal(n)

    return sig.astype(np.float64)


STATES = [
    "normal",
    "bearing_outer",
    "bearing_inner",
    "unbalance",
    "misalignment",
    "looseness",
    "cavitation",
]

STATE_CN = {
    "normal": "正常",
    "bearing_outer": "轴承外圈缺陷",
    "bearing_inner": "轴承内圈缺陷",
    "unbalance": "转子不平衡",
    "misalignment": "联轴器不对中",
    "looseness": "机械松动",
    "cavitation": "汽蚀",
}


if __name__ == "__main__":
    for s in STATES:
        x = generate(s)
        print(f"{s:16s} {STATE_CN[s]:8s} 样点数={len(x):6d}  "
              f"RMS={np.sqrt(np.mean(x**2)):.3f}")
