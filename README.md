# 🧠 XAI-Agent — Make Any ML Model Explain Itself

> **The problem:** Your ML model predicts. Your manager asks *why*. You open a SHAP plot. Their eyes glaze over.
>
> **The solution:** XAI-Agent — an autonomous Hermes Agent pipeline that turns any trained ML model into a plain-English explainability report your entire team can actually read.

---

## 🏆 DEV Hermes Agent Challenge Submission

**Category:** Build With Hermes Agent
**GitHub:** [SimranShaikh20/XAI-Agent](https://github.com/SimranShaikh20)

---

## 🎯 What Problem Does This Solve?

### The XAI Gap — A Real Problem in Every ML Team

Every ML team faces this:

```
Data Scientist  →  "The model uses SHAP values to weight feature contributions"
Product Manager →  "...so should we ship it or not?"
Legal Team      →  "Can you prove it isn't biased?"
Data Scientist  →  😰
```

**Explainable AI tools like SHAP and LIME exist — but they produce:**
- 🔴 Raw numerical arrays no stakeholder understands
- 🔴 Dense technical plots requiring ML expertise to interpret
- 🔴 No bias detection built in
- 🔴 No plain-English narrative
- 🔴 Zero automation — a data scientist must babysit every run

**XAI-Agent solves all of this in one upload.**

---

## ✨ What Makes This Different

| Other XAI Tools | XAI-Agent |
|----------------|-----------|
| Manual Python scripting | Fully autonomous — upload & go |
| SHAP plots (unreadable to most) | Plain-English narrative report |
| No bias detection | Built-in fairness check |
| Single analysis method | Auto-selects best explainer for your model |
| Output for data scientists only | Output for engineers, managers, legal |
| No agentic planning | Hermes Agent 5-step multi-tool pipeline |

---

## 🤖 How Hermes Agent Powers This

XAI-Agent is built entirely around **Hermes Agent's autonomous multi-step planning loop**. Each step is a distinct tool — and each tool's output feeds directly into the next. This is genuine agentic chaining, not a single-shot prompt.

```
┌─────────────────────────────────────────────────────────────┐
│                  HERMES AGENT XAI PIPELINE                  │
├──────┬──────────────────┬─────────────────────────────────── ┤
│ Step │ Tool             │ What it does                        │
├──────┼──────────────────┼─────────────────────────────────── ┤
│  1   │ file_reader      │ Loads model + dataset, auto-detects │
│      │                  │ task type, selects right explainer   │
├──────┼──────────────────┼─────────────────────────────────── ┤
│  2   │ shap_analyzer    │ Runs SHAP across full dataset,      │
│      │                  │ ranks all features by impact +       │
│      │                  │ direction (handles 2D and 3D output) │
├──────┼──────────────────┼─────────────────────────────────── ┤
│  3   │ lime_explainer   │ Picks 3 representative rows,        │
│      │                  │ explains each individual prediction  │
│      │                  │ in plain English                     │
├──────┼──────────────────┼─────────────────────────────────── ┤
│  4   │ bias_checker     │ Scans for demographic features,     │
│      │                  │ flags disparities with evidence      │
├──────┼──────────────────┼─────────────────────────────────── ┤
│  5   │ report_writer    │ Writes full stakeholder report in   │
│      │                  │ Markdown — downloadable instantly    │
└──────┴──────────────────┴─────────────────────────────────── ┘
```

**Why this is genuinely agentic:** The agent doesn't call one tool. It *plans*, executes step-by-step, passes context between tools, handles edge cases (3D SHAP arrays, model type detection, fallback explainers), and produces a coherent final artifact. That is what Hermes Agent is built for.

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/SimranShaikh20/xai-agent
cd xai-agent
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

### 4. Open in browser
```
http://localhost:8501
```

### 5. Upload & run
Use the included `sample_model.pkl` + `sample_dataset.csv`.
Set target column = `target`, task = `Classification`, click **Run explainability analysis**.

---

## 📁 Project Structure

```
xai-agent/
├── app.py                  ← Main Streamlit app (Hermes Agent pipeline)
├── requirements.txt        ← All Python dependencies
├── sample_model.pkl        ← Test model: RandomForest on breast cancer data
├── sample_dataset.csv      ← Test dataset: 569 rows × 30 medical features
└── README.md               ← You are here
```

---

## 🔬 Supported Models

| Framework | Model Types |
|-----------|------------|
| scikit-learn | RandomForest, GradientBoosting, LogisticRegression, SVM, DecisionTree |
| XGBoost | XGBClassifier, XGBRegressor |
| LightGBM | LGBMClassifier, LGBMRegressor |
| Any | Any model saved with `joblib.dump()` or `pickle.dump()` |

**Auto-explainer selection:**
```
Tree-based models  →  SHAP TreeExplainer  (fast, exact)
Everything else    →  SHAP KernelExplainer (universal fallback)
```

---

## 📊 Sample Output

Given the breast cancer dataset (569 patients, 30 tumor measurements):

**Executive Summary**
> This RandomForestClassifier was analyzed across 569 samples and 30 features.
> The single most influential predictor is **'worst area'**.
> No obvious demographic bias was detected in this dataset.

**Top 3 Features**
1. `worst area` — importance: 0.0756 — ↑ increases prediction
2. `worst concave points` — importance: 0.0538 — ↑ increases prediction
3. `mean concave points` — importance: 0.0503 — ↑ increases prediction

**Sample Prediction Explained**
> Row 0 — Predicted: Class 1 (benign) at 94% confidence.
> 'worst area' decreased the malignancy prediction (impact: 0.141).
> 'worst concave points' also decreased it (impact: 0.089).

---

## 🧰 Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent framework | **Hermes Agent** (multi-step planning) |
| Global explainability | **SHAP** (TreeExplainer + KernelExplainer) |
| Local explainability | **LIME** (LimeTabularExplainer) |
| UI | **Streamlit** |
| ML support | scikit-learn, XGBoost, LightGBM |
| Visualization | Matplotlib |
| Report export | Markdown |

---

## 🧪 Generate Your Own Test Model

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
import joblib, pandas as pd

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, "sample_model.pkl")
X.assign(target=y).to_csv("sample_dataset.csv", index=False)
print("Done! Upload both files to XAI-Agent.")
```

---

## 💡 Why This Matters Beyond the Challenge

**Regulatory pressure is real.** The EU AI Act requires high-risk AI systems to be explainable. GDPR gives users the right to explanation for automated decisions. US financial regulators demand model documentation.

**XAI-Agent makes compliance accessible** — not just to ML teams with dedicated explainability engineers, but to any team that can upload a file and click a button.

This is what open-source AI agents should do: take expert-level analysis and make it universally accessible.

---

## 🙏 Built With

- [Hermes Agent](https://hermesagent.dev) — autonomous multi-step agent framework
- [SHAP](https://github.com/slundberg/shap) — SHapley Additive exPlanations
- [LIME](https://github.com/marcotcr/lime) — Local Interpretable Model-agnostic Explanations
- [Streamlit](https://streamlit.io) — Python web app framework

---

## 👤 Author

**Simran Shaikh**
- DEV: [@SimranShaikh20](https://dev.to/simranshaikh20)
- GitHub: [@SimranShaikh20](https://github.com/SimranShaikh20)

