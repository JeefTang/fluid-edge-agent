# 诊断报文契约（冻结）

> 本契约是整套系统的**单一事实源**。诊断模型层输出的 JSON 是唯一权威结论，
> 智能体层（OpenClaw）**只读不改**。字段一经冻结，联调各方不得自行增删。

## 一、报文示例

```json
{
  "schema_version": "1.0",
  "ts": "2026-10-05T16:22:58+08:00",
  "device_id": "ZSB-01-注水A泵",
  "health_score": 62,
  "fault_type": ["bearing_outer"],
  "confidence": 0.87,
  "risk_level": "medium",
  "remaining_life_days": [7, 14],
  "key_evidence": [
    {"metric": "bpfo_prom", "value": 35.1, "baseline": 2.3, "deviation": 14.3,
     "window": "1s窗", "unit": "×中位数"}
  ],
  "suggestion_hint": "check_drive_end_bearing",
  "model_version": "fluid_diag_rulebased_v0.1",
  "degraded": false
}
```

## 二、字段清单（10 必填 + 2 可选）

| 字段 | 类型 | 取值域 | 必填 | 说明 |
|:---|:---|:---|:---:|:---|
| `schema_version` | string | 主版本须为 `1` | ✅ | 主版本不匹配触发兜底 |
| `ts` | string | ISO8601 带时区 | ✅ | 诊断结果生成时刻 |
| `device_id` | string | 与设备档案一致 | ✅ | 话术中转为设备中文名 |
| `health_score` | int | 0~100 整数 | ✅ | 健康度评分 |
| `fault_type` | array | 受控枚举（见下）；正常为 `["normal"]` | ✅ | 允许多值 |
| `confidence` | float | 0.00~1.00 | ✅ | 取最高置信度（多故障取首元素） |
| `risk_level` | enum | normal/low/medium/high/critical | ✅ | 与 health_score 冲突时**以此为准** |
| `remaining_life_days` | array/null | `[下限,上限]` 天；无法估计为 null | ⭕ | null 时禁止输出寿命 |
| `key_evidence` | array | 元素 0~6，见下 | ✅ | 为空时禁止输出"依据"段 |
| `suggestion_hint` | enum | 受控枚举 | ⭕ | 话术模板选择的首要线索 |
| `model_version` | string | 版本号 | ✅ | 写日志与存证，可追溯 |
| `degraded` | bool | true/false | ✅ | true 时进入降级话术前缀 |

> 命名统一：用 `ts`（不用 `timestamp`）、`confidence`（不用 `fault_confidence`）。

## 三、故障类型受控枚举

```
normal            正常
bearing_outer     轴承外圈
bearing_inner     轴承内圈
bearing_ball      滚动体
seal_wear         柱塞密封磨损/泄漏
valve_wear        阀组磨损/敲击异常
cavitation        汽蚀
misalignment      联轴器不对中
unbalance         转子不平衡
looseness         机械松动
overload          过载
clog              堵塞/卡阀
electrical_fault  电机/变频器侧异常
sensor_drift      传感器漂移
unclassified      未分类异常
```

## 四、健康度分档 → 风险等级（5 档）

| health_score | risk_level | 语义 | 默认动作等级 | 人工确认 |
|:---:|:---|:---|:---|:---:|
| 90~100 | normal | 正常 | L0 仅记录 | 否 |
| 75~89 | low | 轻微偏离基线 | L1 记录并观察 | 否 |
| 60~74 | medium | 明确劣化 | L2 自动工单 | 否 |
| 40~59 | high | 严重风险 | L3 建议降负荷＋告警 | ✅ |
| 0~39 | critical | 临近失效 | L4 建议停机（封锁启机） | ✅ 双人 |

> `risk_level` 与 `health_score` 冲突时**以 risk_level 为准**（安全侧优先）。

## 五、证据结构（key_evidence 元素）

| 字段 | 说明 |
|:---|:---|
| `metric` | 指标名（如 `vib_rms` / `bpfo_prom` / `kurtosis`） |
| `value` | 当前值 |
| `baseline` | 基线值（现场标定） |
| `deviation` | 相对基线偏差（无则 null） |
| `window` | 统计窗口（如 `7d` / `1s窗`） |
| `unit` | 单位 |

## 六、本仓库实现说明

`src/diagnose.py` 输出的报文严格遵循本契约；`src/health.py` 中的
`GRADE` / `FAULT_CN` / `HINT` 是该契约在规则诊断器中的落地常量。
