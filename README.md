<!-- SEO & Discovery Keywords:
hermes agent, explainable AI, XAI agent, SHAP LIME automation, ML model explainability,
AI fairness, bias detection, open source AI agent, agentic AI, autonomous ML pipeline,
streamlit machine learning app, model interpretability tool, hermes agent challenge,
feature importance visualization, plain english ML report, EU AI Act compliance
-->

<div align="center">

# 🧠 XAI-Agent — Model Whisperer

### *Make any ML model explain itself. In plain English. Automatically.*

[![Hermes Agent](https://img.shields.io/badge/Powered%20by-Hermes%20Agent-6366f1?style=for-the-badge)](https://hermesagent.dev)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776ab?style=for-the-badge&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![DEV Challenge](https://img.shields.io/badge/DEV-Hermes%20Agent%20Challenge-0a0a0a?style=for-the-badge)](https://dev.to/challenges/hermes-agent)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

<br/>

> **The gap nobody talks about:**
> Data scientists speak SHAP. Managers speak English. Legal speaks risk.
> XAI-Agent is the autonomous bridge — powered by Hermes Agent's multi-step planning pipeline.

<br/>

**[🚀 Quick Start](#-quick-start) · [🤖 How It Works](#-how-hermes-agent-powers-this) · [📊 Sample Output](#-real-output-example) · [🎯 Why This Wins](#-why-this-matters)**

</div>

---

## 🔥 The Problem This Solves

Every ML team has this meeting:

```
Data Scientist  →  "The SHAP values show mean concave points has 0.0503 importance"
Product Manager →  "...great. Should we ship it or not?"
Legal Counsel   →  "Is it biased against any demographic group?"
Data Scientist  →  "I'll need 3 days to write that report."
Manager         →  "We needed it yesterday."
```

**This happens everywhere — at startups, banks, hospitals, and governments — every single day.**

Explainable AI (XAI) tools like SHAP and LIME have existed for years. But they require:
- A data scientist to write custom analysis code for every model
- ML expertise to interpret the outputs
- Hours of manual work to turn plots into stakeholder reports
- A separate fairness audit for bias detection
- Yet another tool to generate a downloadable document

**There is no single tool that does all of this autonomously. Until now.**

---

## ✅ What XAI-Agent Does

Upload your trained model + dataset. Click one button. Get back:

| Output | What it means |
|--------|--------------|
| **Executive Summary** | 3-sentence plain-English overview for your CEO |
| **SHAP Feature Importance Chart** | Visual ranking of what drives every prediction |
| **Top 5 Features Explained** | Exactly which inputs matter and why, in human language |
| **3 Predictions Explained** | Why the model made *that specific decision* for *that specific row* |
| **Bias & Fairness Badge** | ✅ green or ⚠️ amber — with evidence, not just a guess |
| **Downloadable Markdown Report** | Ready to paste into Confluence, Notion, or a PR |

**Total time: under 3 minutes. Zero code required after setup.**

---

## ✨ How XAI-Agent Stands Apart

| Capability | Traditional XAI Workflow | XAI-Agent |
|-----------|--------------------------|-----------|
| Setup | Write custom Python per model | Upload 2 files |
| Explainer selection | Manual — must know SHAP internals | Auto-detects model type, picks best explainer |
| Output audience | Data scientists only | Everyone — engineers, managers, legal, regulators |
| Bias detection | Separate tool / manual audit | Built into the pipeline, runs automatically |
| SHAP version compatibility | Often breaks on 3D output arrays | Handles 2D and 3D SHAP arrays natively |
| Report format | Raw plots + Jupyter notebooks | Structured plain-English Markdown report |
| Automation level | Single-shot scripts | Hermes Agent 5-tool autonomous planning loop |
| Regulatory alignment | Not considered | EU AI Act + GDPR explainability requirements in mind |

---

## 🤖 How Hermes Agent Powers This

XAI-Agent is **not** a wrapper around a single function call. It is a genuine **Hermes Agent autonomous pipeline** — five tools, each with a specific responsibility, chained together so each step's output becomes the next step's input.

This is exactly the kind of multi-step, multi-tool reasoning that Hermes Agent is built for.

```
╔══════════════════════════════════════════════════════════════════╗
║              HERMES AGENT — XAI AUTONOMOUS PIPELINE             ║
╠═══════╦══════════════════════╦═══════════════════════════════════╣
║ Step  ║ Tool                 ║ Responsibility                    ║
╠═══════╬══════════════════════╬═══════════════════════════════════╣
║  1/5  ║ file_reader          ║ Loads .pkl model + .csv dataset   ║
║       ║                      ║ Auto-detects: classification vs   ║
║       ║                      ║ regression. Flags class imbalance ║
║       ║                      ║ Selects right SHAP explainer type ║
╠═══════╬══════════════════════╬═══════════════════════════════════╣
║  2/5  ║ shap_analyzer        ║ Runs SHAP on full dataset sample  ║
║       ║                      ║ TreeExplainer for tree models     ║
║       ║                      ║ KernelExplainer as fallback       ║
║       ║                      ║ Handles 2D + 3D SHAP output arrays║
║       ║                      ║ Ranks all features by mean |SHAP| ║
║       ║                      ║ Computes direction: ↑ or ↓        ║
╠═══════╬══════════════════════╬═══════════════════════════════════╣
║  3/5  ║ lime_explainer       ║ Selects 3 representative samples  ║
║       ║                      ║ (one per class for classifiers)   ║
║       ║                      ║ Runs LIME on each instance        ║
║       ║                      ║ Converts weights to plain English ║
╠═══════╬══════════════════════╬═══════════════════════════════════╣
║  4/5  ║ bias_checker         ║ Scans feature names for sensitive ║
║       ║                      ║ attributes: age, gender, race,    ║
║       ║                      ║ income, zip, religion, ethnicity  ║
║       ║                      ║ Checks prediction class imbalance ║
║       ║                      ║ Returns evidence-backed verdict   ║
╠═══════╬══════════════════════╬═══════════════════════════════════╣
║  5/5  ║ report_writer        ║ Assembles all tool outputs into   ║
║       ║                      ║ structured Markdown report        ║
║       ║                      ║ Writes executive summary, feature ║
║       ║                      ║ table, LIME explanations, bias    ║
║       ║                      ║ section, and recommendations      ║
╚═══════╩══════════════════════╩═══════════════════════════════════╝
```

### Why This Is Genuinely Agentic

Most "AI agent" demos call one LLM endpoint and display the result. XAI-Agent does something fundamentally different:

**Context flows between tools.** The model type detected in Step 1 determines which SHAP explainer Step 2 uses. The SHAP feature ranking from Step 2 informs which features Step 3's LIME focuses on. The bias verdict from Step 4 shapes the recommendations in Step 5's report. Every tool is aware of what came before.

**The agent handles failure gracefully.** If TreeExplainer fails, the agent falls back to KernelExplainer automatically — no crash, no error shown to the user. This is what production-grade agentic systems do.

**Real edge cases are handled.** Newer versions of SHAP return 3D arrays `(samples, features, classes)` instead of 2D. The agent detects this and slices `[:, :, 1]` automatically — a bug that breaks every naive SHAP implementation.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip

### 1 — Clone
```bash
git clone https://github.com/SimranShaikh20/xai-agent.git
cd xai-agent
```

### 2 — Install
```bash
pip install -r requirements.txt
```
> First install takes ~2 minutes (SHAP and LightGBM are large). Subsequent runs are instant.

### 3 — Run
```bash
streamlit run app.py
```

### 4 — Open browser
```
http://localhost:8501
```

### 5 — Test with included files
Upload `sample_model.pkl` + `sample_dataset.csv` → set target column = `target` → click **Run explainability analysis**.

Full analysis completes in under 3 minutes.

---

## 📊 Real Output Example

Running XAI-Agent on the included breast cancer dataset (569 patients, 30 tumor measurements, RandomForestClassifier):

### Executive Summary
> This **RandomForestClassifier** was analyzed across **569 samples** and **30 features**.
> The single most influential predictor is **'worst area'** — tumors with larger worst-case area measurements are significantly more likely to be predicted as malignant.
> **No obvious demographic bias was detected** — the dataset contains no sensitive demographic features.

### Top 5 Features by SHAP Importance

| Rank | Feature | SHAP Score | Effect |
|------|---------|-----------|--------|
| 1 | worst area | 0.0756 | ↑ increases malignancy prediction |
| 2 | worst concave points | 0.0538 | ↑ increases malignancy prediction |
| 3 | mean concave points | 0.0503 | ↑ increases malignancy prediction |
| 4 | worst perimeter | 0.0489 | ↑ increases malignancy prediction |
| 5 | worst radius | 0.0401 | ↑ increases malignancy prediction |

### Sample Prediction Explained (Plain English)
> **Row 0 — Predicted: Class 1 (benign) at 94% confidence**
>
> The model is highly confident this tumor is benign. The main reason:
> **'worst area'** was well below the threshold that typically signals malignancy — this single feature decreased the malignancy prediction by 0.141 points.
> **'worst concave points'** also supported the benign classification (impact: −0.089).

### Bias & Fairness
> ✅ **No demographic bias detected.** The dataset contains no features named age, gender, sex, race, ethnicity, zip code, income, religion, or nationality. Prediction distribution: 62.7% benign, 37.3% malignant — within acceptable range.

---

## 📁 Project Structure

```
xai-agent/
│
├── app.py                    ← Streamlit app + full Hermes Agent pipeline
│   ├── HermesXAIAgent class  ← Agent with 5 autonomous tools
│   ├── tool_file_reader()    ← Model + dataset loader
│   ├── tool_shap_analyzer()  ← Global SHAP explainability
│   ├── tool_lime_explainer() ← Local LIME explanations
│   ├── tool_bias_checker()   ← Fairness & demographic audit
│   └── tool_report_writer()  ← Plain-English report generator
│
├── requirements.txt          ← Pinned Python dependencies
├── generate_test_data.py     ← Script to regenerate test files
├── sample_model.pkl          ← Pre-trained RandomForest (breast cancer)
├── sample_dataset.csv        ← 569 rows × 31 cols (30 features + target)
├── XAI_Agent_Hermes.ipynb    ← Google Colab version (zero install)
├── .gitignore
└── README.md
```

---

## 🔬 Supported Models & Auto-Explainer Logic

```python
# The agent auto-selects the right SHAP explainer:

TREE_MODELS = [
    "RandomForestClassifier",   "RandomForestRegressor",
    "GradientBoostingClassifier","GradientBoostingRegressor",
    "XGBClassifier",            "XGBRegressor",
    "LGBMClassifier",           "LGBMRegressor",
    "DecisionTreeClassifier",   "ExtraTreesClassifier"
]

if model_type in TREE_MODELS:
    explainer = shap.TreeExplainer(model)   # Fast, exact, recommended
else:
    explainer = shap.KernelExplainer(model) # Universal fallback
```

**SHAP version compatibility fix (unique to this project):**
```python
# New SHAP versions return 3D arrays: (samples, features, classes)
# This breaks every naive implementation. XAI-Agent handles it:

if isinstance(shap_raw, list):
    shap_vals = shap_raw[1]              # Old SHAP: list per class
elif shap_raw.ndim == 3:
    shap_vals = shap_raw[:, :, 1]       # New SHAP: 3D → slice class 1
else:
    shap_vals = shap_raw                 # Regression: use as-is
```

---

## 🧰 Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| **Agent framework** | Hermes Agent | Multi-step autonomous planning |
| **Global XAI** | SHAP 0.44+ | Industry standard, model-agnostic |
| **Local XAI** | LIME 0.2+ | Instance-level explanations |
| **UI** | Streamlit 1.32+ | Fast, Python-native, no JS needed |
| **ML frameworks** | scikit-learn, XGBoost, LightGBM | Covers 95% of real-world models |
| **Visualization** | Matplotlib | Reliable, exportable charts |
| **Report** | Markdown | Portable — works in GitHub, Notion, Confluence |

---

## 🧪 Use Your Own Model

```python
# Step 1: Train any model
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
import joblib, pandas as pd

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Step 2: Save it
joblib.dump(model, "my_model.pkl")
X.assign(target=y).to_csv("my_dataset.csv", index=False)

# Step 3: Upload both files to XAI-Agent → done
```

Works with any model savedvia `joblib.dump()` or `pickle.dump()`.

---

## 💡 Why This Matters Beyond the Challenge

### Regulatory Pressure Is Real and Growing

**EU AI Act (2024):** High-risk AI systems must provide meaningful explanations for automated decisions. Non-compliance: up to €30 million or 6% of global revenue.

**GDPR Article 22:** EU citizens have the right to explanation when subject to automated decision-making. Organizations must be able to explain *why* a model reached a decision.

**US Financial Regulation:** The Equal Credit Opportunity Act requires lenders to explain adverse actions. Banks using ML credit scoring need model explainability built in.

**Healthcare AI:** FDA guidance on AI/ML-based software as a medical device (SaMD) requires transparency about how models make decisions.

### Who Needs This Right Now

- **ML engineers** shipping models to production — need audit trails
- **Data scientists** presenting to non-technical stakeholders — need plain English
- **Compliance teams** at banks, hospitals, insurance companies — need bias evidence
- **Product managers** evaluating ML features — need to understand model behavior
- **Open-source contributors** — need a free alternative to expensive XAI platforms like Fiddler, Arize, or Arthur AI (which cost $50K+/year)

**XAI-Agent is free, open-source, runs locally, and works in 3 minutes.**

---

## 🔮 What's Next (Roadmap)

- [ ] PDF report export with charts embedded
- [ ] Support for PyTorch and TensorFlow neural networks (DeepExplainer)
- [ ] SHAP interaction values for feature pair analysis
- [ ] Counterfactual explanations ("what would need to change to flip this prediction?")
- [ ] Batch analysis across multiple models for comparison
- [ ] REST API mode for CI/CD pipeline integration
- [ ] Streamlit Cloud one-click deployment

---

## 🙏 Acknowledgements

- [Hermes Agent](https://hermesagent.dev) — the autonomous agent framework that makes the pipeline possible
- [SHAP](https://github.com/slundberg/shap) by Scott Lundberg — the gold standard for ML explainability
- [LIME](https://github.com/marcotcr/lime) by Marco Tulio Ribeiro — local model-agnostic explanations
- [Streamlit](https://streamlit.io) — for making Python web apps actually enjoyable to build
- The [DEV Community](https://dev.to) for running the Hermes Agent Challenge

---

## 👤 Author

**Simran Shaikh** — AI/ML Engineer

- 🐦 DEV: [@SimranShaikh20](https://dev.to/simranshaikh20)
- 💻 GitHub: [@SimranShaikh20](https://github.com/SimranShaikh20)

---

<div align="center">

**Built for the [DEV Hermes Agent Challenge](https://dev.to/challenges/hermes-agent) — May 2026**

*If this helped you, drop a ⭐ on GitHub and a ❤️ on the DEV post — it means a lot!*

</div>