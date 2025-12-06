# Zitto-Liion-Abuse250 Dataset
**Synthetic Electro-Thermal Abuse Simulation Dataset (6 Scenarios × 250 Runs)**  
**Version:** 1.0  
**License:** CC BY-NC 4.0 (recommended)

Zitto-Liion-Abuse250 is a high-fidelity synthetic dataset containing **1500 electro-thermal abuse simulations** of a lithium-ion cell generated using the **DFN (Doyle–Fuller–Newman) model** with **Chen2020** parameters.  
Each of the **six scenarios** includes **250 randomized runs**, providing a compact but diverse dataset tailored for **battery failure prediction**, **early-warning modeling**, and **multi-signal scenario classification**.

This dataset is designed for researchers building **data-driven battery safety models**, especially **short-horizon thermal runaway prediction** using LSTMs, BiLSTMs, transformers, and hybrid architectures.

---

## 📦 Dataset Contents

Each run is exported as a standalone CSV containing:

- `time_s` — simulation time  
- `current_A` — applied current  
- `voltage_V` — terminal voltage  
- `temperature_C` — volume-averaged cell temperature  
- `temperature_C_runaway` — synthetic runaway proxy  
- `dT_dt_C_per_s` — temperature derivative  
- `isc_active` — internal short indicator  
- `external_heating_active` — external heat source indicator  
- `decomposition_active` — decomposition onset flag  
- `danger_flag` — binary high-severity indicator  
- `scenario` — abuse mode label  
- Scenario metadata: C-rate, over-factor, depth-factor, heating rate, SOC label, etc.  

Total size: **1500 runs** (~229,000 timesteps).  

---

## 🔥 Abuse & Nominal Scenarios (6 Modes)

| Mode | Description |
|------|-------------|
| **Overcharge** | Random C-rate and overcharge factor; varied cooling; voltage-driven abuse. |
| **Overdischarge** | Deep discharge with randomized depth factor; low-voltage stress. |
| **Thermal\_abuse** | Oven-like heating ramp (~+150°C); external heating flag active. |
| **ISC (Internal Short)** | Forced short-circuit; monotonically increasing thermal response. |
| **Combined** | Overcharge + external heating; hybrid electro-thermal stress. |
| **Normal\_cycle** | Safe charging/discharging; no runaway; provides benign baseline. |

Parameter ranges are randomized per run to yield realistic variability.

---

## 🌡️ Synthetic Runaway Model

To emulate localized hot-spot escalation (difficult to capture directly in lumped thermal DFN models), the dataset includes a **synthetic runaway channel**:

- When temperature exceeds a trigger threshold,  
  an additional self-heating term is added.  
- Runaway temperature is **clamped at 300°C**, preventing numerical blow-ups while mimicking realistic post-runaway upper limits.  
- This proxy is **not** a literal physical temperature, but a stable ML-friendly indicator of thermal escalation.

---

## 🧪 Labeling and Risk Scoring

Each timestep is assigned a continuous **risk score**:
\[
\text{risk\_score}(t)=\mathbf{w}^\top \mathbf{s}(t)
\]

and discretized into:

- **SAFE** (< 0.33)  
- **WARNING** (0.33–0.66) — intentionally narrow due to fast-onset abuse dynamics  
- **DANGER** (≥ 0.66)

**Global distribution:**  
SAFE: 51.3% — WARNING: 2.7% — DANGER: 46.0%

The WARNING band is narrow because thermal runaway in several scenarios (ISC, combined, thermal abuse) transitions rapidly from stable to high-severity behavior.

---

## 🪟 Sliding Windows for ML Training

To emulate real-time early-warning systems:

- Window length: **32**  
- Stride: **4**  
- Horizon: **10** (predict “danger within next 10 steps?”)

For every window:
- `y_fail`: horizon-based failure label  
- `y_mode`: one of the six abuse modes  

---

## 🧠 Example Usage (Python)

```python
import pandas as pd

df = pd.read_csv("data/overcharge/run_0001.csv")
print(df.head())
