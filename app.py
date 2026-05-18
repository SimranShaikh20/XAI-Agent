import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import joblib
import time
import io
import base64
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="XAI-Agent — Model Whisperer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CUSTOM CSS — matches Lovable UI exactly ──────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Reset & base */
*, *::before, *::after { box-sizing: border-box; }
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif;
    background: #FFFFFF;
    color: #0F172A;
}
[data-testid="stAppViewContainer"] { background: #FFFFFF; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stSidebar"] { display: none; }

/* Hide streamlit defaults */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── TOP NAVBAR ── */
.navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 32px;
    border-bottom: 1px solid #E2E8F0;
    background: #FFFFFF;
    position: sticky;
    top: 0;
    z-index: 100;
}
.navbar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 18px;
    font-weight: 700;
    color: #0F172A;
}
.navbar-brand .logo-icon {
    width: 34px; height: 34px;
    background: linear-gradient(135deg, #1a56db, #7c3aed);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    color: white; font-size: 16px;
}
.navbar-brand span.blue { color: #1a56db; }
.navbar-right {
    display: flex;
    align-items: center;
    gap: 16px;
}
.navbar-status {
    font-size: 13px;
    color: #64748B;
}
.btn-new-analysis {
    background: #1a56db;
    color: white;
    border: none;
    padding: 8px 16px;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    display: flex; align-items: center; gap: 6px;
}

/* ── MAIN LAYOUT ── */
.main-layout {
    display: grid;
    grid-template-columns: 1fr 300px;
    gap: 0;
    min-height: calc(100vh - 64px);
    padding: 40px 32px;
    max-width: 1200px;
    margin: 0 auto;
}
.main-left { padding-right: 40px; }
.main-right {
    border-left: 1px solid #E2E8F0;
    padding-left: 28px;
}

/* ── HERO TEXT ── */
.hero-title {
    font-size: 42px;
    font-weight: 800;
    color: #0F172A;
    line-height: 1.1;
    margin-bottom: 12px;
}
.hero-title .green { color: #059669; }
.hero-subtitle {
    font-size: 15px;
    color: #64748B;
    margin-bottom: 32px;
    max-width: 580px;
    line-height: 1.6;
}

/* ── UPLOAD ZONE ── */
.upload-zone {
    border: 2px dashed #CBD5E1;
    border-radius: 12px;
    padding: 48px 32px;
    text-align: center;
    background: #F8FAFC;
    margin-bottom: 20px;
    transition: all 0.2s;
}
.upload-zone:hover { border-color: #1a56db; background: #EFF6FF; }
.upload-icon { font-size: 32px; margin-bottom: 12px; }
.upload-title { font-size: 15px; font-weight: 500; color: #0F172A; margin-bottom: 6px; }
.upload-hint { font-size: 12px; color: #94A3B8; margin-bottom: 20px; }

/* ── FILE CHIPS ── */
.file-chips { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 24px; }
.file-chip {
    display: flex; align-items: center; gap: 10px;
    background: #F1F5F9;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 10px 14px;
    font-size: 13px;
}
.chip-icon {
    width: 32px; height: 32px;
    border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px;
}
.chip-icon.model { background: #DBEAFE; }
.chip-icon.data  { background: #D1FAE5; }
.chip-name { font-weight: 500; color: #0F172A; }
.chip-meta { font-size: 11px; color: #94A3B8; }

/* ── SETTINGS PANEL ── */
.settings-title {
    font-size: 15px; font-weight: 600; color: #0F172A;
    margin-bottom: 20px;
}
.settings-label {
    font-size: 11px; font-weight: 600;
    color: #64748B; letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 10px;
}

/* Task type buttons */
.task-btns { display: flex; gap: 6px; margin-bottom: 24px; }
.task-btn {
    flex: 1; padding: 8px 4px;
    border-radius: 8px; border: 1px solid #E2E8F0;
    background: #F8FAFC; color: #64748B;
    font-size: 12px; font-weight: 500;
    cursor: pointer; text-align: center;
    font-family: 'Inter', sans-serif;
}
.task-btn.active {
    background: #1a56db; color: white;
    border-color: #1a56db;
}

/* ── RUN BUTTON ── */
.run-btn {
    width: 100%;
    background: linear-gradient(135deg, #1a56db, #7c3aed);
    color: white; border: none;
    padding: 13px 20px;
    border-radius: 10px;
    font-size: 14px; font-weight: 600;
    cursor: pointer;
    display: flex; align-items: center;
    justify-content: center; gap: 8px;
    margin-bottom: 12px;
    font-family: 'Inter', sans-serif;
}
.run-btn:disabled { opacity: 0.5; cursor: not-allowed; }
.run-btn-hint {
    font-size: 11px; color: #94A3B8;
    display: flex; align-items: flex-start;
    gap: 6px; line-height: 1.5;
}

/* ── AGENT STEPS ── */
.agent-steps { margin: 28px 0; }
.agent-steps-title {
    font-size: 13px; font-weight: 600;
    color: #0F172A; margin-bottom: 14px;
    display: flex; align-items: center; gap: 8px;
}
.step-item {
    display: flex; align-items: flex-start;
    gap: 12px; padding: 10px 0;
    border-bottom: 1px solid #F1F5F9;
}
.step-dot {
    width: 24px; height: 24px; border-radius: 50%;
    display: flex; align-items: center;
    justify-content: center; font-size: 11px;
    flex-shrink: 0; margin-top: 1px;
    font-weight: 600;
}
.step-dot.done  { background: #D1FAE5; color: #059669; }
.step-dot.running { background: #DBEAFE; color: #1a56db; }
.step-dot.pending { background: #F1F5F9; color: #94A3B8; }
.step-text { font-size: 13px; }
.step-name { font-weight: 500; color: #0F172A; }
.step-desc { font-size: 12px; color: #94A3B8; margin-top: 1px; }

/* ── REPORT SECTIONS ── */
.report-wrap { margin-top: 32px; }
.report-section {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.report-section-title {
    font-size: 14px; font-weight: 600;
    color: #0F172A; margin-bottom: 12px;
    display: flex; align-items: center; gap: 8px;
}
.exec-summary {
    background: linear-gradient(135deg, #EFF6FF, #F0FDF4);
    border: 1px solid #BFDBFE;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.exec-summary-title {
    font-size: 13px; font-weight: 700;
    color: #1a56db; text-transform: uppercase;
    letter-spacing: 0.08em; margin-bottom: 10px;
}
.exec-summary-text {
    font-size: 14px; color: #0F172A;
    line-height: 1.7;
}

/* Feature bar */
.feat-row { margin-bottom: 10px; }
.feat-label {
    display: flex; justify-content: space-between;
    font-size: 12px; color: #0F172A;
    margin-bottom: 4px;
}
.feat-bar-bg {
    background: #E2E8F0; border-radius: 4px; height: 8px;
}
.feat-bar-fill {
    height: 8px; border-radius: 4px;
    background: linear-gradient(90deg, #1a56db, #059669);
}

/* Bias badge */
.bias-ok   { background: #D1FAE5; color: #065F46; padding: 6px 14px; border-radius: 20px; font-size: 13px; font-weight: 500; display: inline-flex; align-items: center; gap: 6px; }
.bias-warn { background: #FEF3C7; color: #92400E; padding: 6px 14px; border-radius: 20px; font-size: 13px; font-weight: 500; display: inline-flex; align-items: center; gap: 6px; }

/* Download btn */
.dl-btn {
    display: inline-flex; align-items: center; gap: 8px;
    background: #0F172A; color: white;
    padding: 10px 20px; border-radius: 8px;
    font-size: 13px; font-weight: 500;
    text-decoration: none; margin-right: 10px;
    margin-top: 8px;
}
.dl-btn.outline {
    background: transparent; color: #0F172A;
    border: 1px solid #E2E8F0;
}

/* Footer note */
.footer-note {
    font-size: 12px; color: #94A3B8;
    margin-top: 24px; line-height: 1.6;
    padding-top: 16px; border-top: 1px solid #F1F5F9;
}

/* Streamlit widget overrides */
.stFileUploader > div { border: none !important; background: transparent !important; padding: 0 !important; }
.stFileUploader label { display: none !important; }
div[data-testid="stFileUploaderDropzone"] {
    border: 2px dashed #CBD5E1 !important;
    border-radius: 12px !important;
    background: #F8FAFC !important;
    padding: 32px !important;
}
.stTextInput > div > div > input {
    border-radius: 8px !important;
    border: 1px solid #E2E8F0 !important;
    font-size: 13px !important;
    padding: 10px 14px !important;
}
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #1a56db, #7c3aed) !important;
}
</style>
""", unsafe_allow_html=True)


# ── HERMES AGENT CLASS ────────────────────────────────────────────────────────
class HermesXAIAgent:
    """
    Hermes Agent — XAI Pipeline
    Multi-step autonomous agent for ML model explainability
    """

    STEPS = [
        ("Inspect",        "Load model & dataset, detect task type"),
        ("Global SHAP",    "Compute SHAP values across full dataset"),
        ("Local LIME",     "Explain 3 representative predictions"),
        ("Bias Check",     "Scan for demographic disparities"),
        ("Report",         "Write plain-English explainability report"),
    ]

    def __init__(self):
        self.results = {}

    # TOOL 1 — file_reader
    def tool_file_reader(self, model_bytes, dataset_bytes, target_col):
        import sklearn
        model = joblib.load(io.BytesIO(model_bytes))
        df    = pd.read_csv(io.BytesIO(dataset_bytes))
        feat_cols = [c for c in df.columns if c != target_col]
        X = df[feat_cols]
        y = df[target_col] if target_col in df.columns else None

        task = "classification"
        if y is not None and y.nunique() > 10:
            task = "regression"

        model_type = type(model).__name__
        self.results.update({
            "model": model, "X": X, "y": y, "df": df,
            "feat_cols": feat_cols, "model_type": model_type, "task": task
        })
        return model_type, len(feat_cols), len(df), task

    # TOOL 2 — shap_analyzer
    def tool_shap_analyzer(self):
        import shap
        model      = self.results["model"]
        X          = self.results["X"]
        feat_cols  = self.results["feat_cols"]
        model_type = self.results["model_type"]

        X_sample = X.sample(min(150, len(X)), random_state=42)

        tree_models = [
            "RandomForestClassifier","RandomForestRegressor",
            "GradientBoostingClassifier","GradientBoostingRegressor",
            "XGBClassifier","XGBRegressor",
            "LGBMClassifier","LGBMRegressor",
            "DecisionTreeClassifier","ExtraTreesClassifier"
        ]

        if model_type in tree_models:
            explainer = shap.TreeExplainer(model)
            shap_raw  = explainer.shap_values(X_sample)
            # Handle all shap output shapes:
            # list of arrays (old shap, per class) → take class 1
            # 3D array (samples, features, classes) → take class 1
            # 2D array (samples, features) → use as-is
            if isinstance(shap_raw, list):
                shap_vals = shap_raw[1] if len(shap_raw) > 1 else shap_raw[0]
            elif hasattr(shap_raw, 'ndim') and shap_raw.ndim == 3:
                shap_vals = shap_raw[:, :, 1]
            else:
                shap_vals = shap_raw
            method = "SHAP TreeExplainer"
        else:
            bg = shap.sample(X_sample, 40)
            explainer = shap.KernelExplainer(model.predict_proba, bg)
            shap_raw  = explainer.shap_values(X_sample.iloc[:40])
            if isinstance(shap_raw, list):
                shap_vals = shap_raw[1] if len(shap_raw) > 1 else shap_raw[0]
            elif hasattr(shap_raw, 'ndim') and shap_raw.ndim == 3:
                shap_vals = shap_raw[:, :, 1]
            else:
                shap_vals = shap_raw
            method = "SHAP KernelExplainer"

        shap_vals = np.array(shap_vals)
        mean_shap = np.abs(shap_vals).mean(axis=0)
        directions = [
            "↑ positive" if float(shap_vals[:, i].mean()) > 0 else "↓ negative"
            for i in range(len(feat_cols))
        ]
        imp_df = pd.DataFrame({
            "feature": feat_cols,
            "importance": mean_shap,
            "direction": directions
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        # SHAP bar chart
        fig, ax = plt.subplots(figsize=(9, 5))
        top10 = imp_df.head(10)
        colors = ["#1a56db" if d == "↑ positive" else "#059669" for d in top10["direction"]]
        ax.barh(top10["feature"][::-1], top10["importance"][::-1], color=colors[::-1])
        ax.set_xlabel("Mean |SHAP value|", fontsize=11)
        ax.set_title("Feature Importance (SHAP)", fontsize=13, fontweight="bold", pad=12)
        ax.spines[["top","right"]].set_visible(False)
        ax.tick_params(labelsize=9)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        plt.close()

        self.results.update({
            "imp_df": imp_df, "shap_vals": shap_vals,
            "X_sample": X_sample, "shap_plot": buf,
            "explainer_method": method
        })
        return imp_df, buf

    # TOOL 3 — lime_explainer
    def tool_lime_explainer(self):
        import lime.lime_tabular
        model     = self.results["model"]
        X         = self.results["X"]
        y         = self.results["y"]
        feat_cols = self.results["feat_cols"]
        task      = self.results["task"]

        explainer = lime.lime_tabular.LimeTabularExplainer(
            X.values, feature_names=feat_cols,
            class_names=["Class 0","Class 1"] if task=="classification" else None,
            mode=task, random_state=42
        )

        # 3 representative samples
        if task == "classification" and y is not None:
            indices = []
            for cls in y.unique()[:2]:
                indices.append(y[y == cls].index[0])
            if len(indices) < 3:
                indices.append(y.index[len(y)//2])
        else:
            n = len(X)
            indices = [0, n//2, n-1]

        predict_fn = model.predict_proba if task=="classification" else model.predict
        lime_results = []

        for idx in indices[:3]:
            instance = X.loc[idx].values
            exp = explainer.explain_instance(
                instance, predict_fn, num_features=5, num_samples=300
            )
            exp_list = exp.as_list()
            if task == "classification":
                proba = model.predict_proba([instance])[0]
                pred  = model.predict([instance])[0]
                label = f"Class {int(pred)} — {proba[int(pred)]:.0%} confidence"
            else:
                pred  = model.predict([instance])[0]
                label = f"Predicted: {pred:.2f}"

            plain = self._to_english(exp_list[:3], label)
            lime_results.append({
                "row": idx, "label": label,
                "reasons": exp_list[:5], "plain": plain
            })

        self.results["lime_results"] = lime_results
        return lime_results

    def _to_english(self, exp_list, label):
        parts = []
        for feat, w in exp_list[:3]:
            verb = "increased" if w > 0 else "decreased"
            parts.append(f"**{feat}** {verb} the prediction (impact: {abs(w):.3f})")
        return f"**{label}** — " + "; ".join(parts)

    # TOOL 4 — bias_checker
    def tool_bias_checker(self):
        feat_cols = self.results["feat_cols"]
        model     = self.results["model"]
        X         = self.results["X"]
        y         = self.results["y"]

        SENSITIVE = ["age","gender","sex","race","ethnicity",
                     "zip","income","religion","nationality"]
        found = [f for f in feat_cols if any(k in f.lower() for k in SENSITIVE)]

        preds = model.predict(X)
        pred_dist = pd.Series(preds).value_counts(normalize=True).to_dict()

        if found:
            flag = True
            msg  = f"⚠️ Sensitive features found: {', '.join(found)}. A fairness audit is recommended before production use."
        else:
            flag = False
            msg  = "✅ No common demographic features (age, gender, race, income, zip) found in this dataset."

        if len(pred_dist) == 2:
            vals = list(pred_dist.values())
            if abs(vals[0]-vals[1]) > 0.3:
                msg += f" Note: model shows {abs(vals[0]-vals[1]):.0%} prediction imbalance between classes."

        self.results.update({"bias_flag": flag, "bias_msg": msg, "pred_dist": pred_dist})
        return flag, msg

    # TOOL 5 — report_writer
    def tool_report_writer(self):
        model_type = self.results["model_type"]
        imp_df     = self.results["imp_df"]
        lime_res   = self.results["lime_results"]
        bias_flag  = self.results["bias_flag"]
       
        bias_msg   = self.results["bias_msg"]
        feat_cols  = self.results["feat_cols"]
        X          = self.results["X"]
        method     = self.results["explainer_method"]
        top_feat   = imp_df.iloc[0]["feature"]

        MODEL_DESC = {
            "RandomForestClassifier": "A Random Forest builds hundreds of decision trees and combines their votes — like asking 100 experts and going with the majority.",
            "GradientBoostingClassifier": "Gradient Boosting builds trees sequentially, each correcting errors of the last — like a team where every member fixes what the previous one got wrong.",
            "LogisticRegression": "Logistic Regression draws a decision boundary between classes using a weighted combination of features.",
            "XGBClassifier": "XGBoost is a high-performance tree ensemble that learns iteratively, correcting its mistakes with each new tree.",
            "DecisionTreeClassifier": "A Decision Tree asks a series of yes/no questions about features, following branches to reach a prediction.",
            "SVC": "A Support Vector Machine finds the optimal boundary that maximally separates classes in feature space.",
        }
        model_desc = MODEL_DESC.get(model_type,
            f"{model_type} is a machine learning model that learns patterns from training data to predict outcomes on new inputs.")

        md = f"""# 🧠 XAI-Agent Explainability Report
*Powered by Hermes Agent*

---

## 1. Executive Summary
This report explains how a **{model_type}** makes its predictions across **{len(X)} samples** and **{len(feat_cols)} features**.
The single most influential factor is **'{top_feat}'**.
Bias check: {"⚠️ Review recommended" if bias_flag else "✅ No obvious bias detected"}.

---

## 2. How This Model Works
{model_desc}

---

## 3. What Drives Predictions (Top 10 Features)

| Rank | Feature | SHAP Score | Effect |
|------|---------|-----------|--------|
"""
        for i, row in imp_df.head(10).iterrows():
            md += f"| {i+1} | {row['feature']} | {row['importance']:.4f} | {row['direction']} |\n"

        md += "\n---\n\n## 4. Sample Predictions Explained\n\n"
        for i, r in enumerate(lime_res, 1):
            md += f"### Sample {i} (Row {r['row']})\n"
            md += f"{r['plain']}\n\n"
            md += "**Top factors:**\n"
            for feat, w in r["reasons"][:3]:
                arrow = "🟢 +" if w > 0 else "🔴 -"
                md += f"- {arrow} `{feat}` → {w:+.3f}\n"
            md += "\n"

        md += f"""---

## 5. Bias & Fairness Assessment
{bias_msg}

---

## 6. Recommendations
- Monitor **'{top_feat}'** closely — it has the highest influence
- Re-run this analysis after every model retraining
- {"Conduct a formal fairness audit before deployment" if bias_flag else "Continue monitoring prediction distributions over time"}

---

## 7. Technical Appendix
- **Explainability method:** {method}
- **Model type:** {model_type}
- **Features analyzed:** {len(feat_cols)}
- **Samples used for SHAP:** {len(self.results["X_sample"])}
- **Confidence level:** High — sufficient data with a tree-based model
"""
        self.results["report_md"] = md
        return md


# ── SESSION STATE ─────────────────────────────────────────────────────────────
if "step_status" not in st.session_state:
    st.session_state.step_status = ["pending"] * 5
if "report_done" not in st.session_state:
    st.session_state.report_done = False
if "agent_results" not in st.session_state:
    st.session_state.agent_results = {}
if "task_type" not in st.session_state:
    st.session_state.task_type = "Classification"
if "navbar_status" not in st.session_state:
    st.session_state.navbar_status = "Awaiting upload"


# ── NAVBAR ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="navbar">
  <div class="navbar-brand">
    <div class="logo-icon">◈</div>
    <span>XAI<span class="blue">-Agent</span></span>
  </div>
  <div class="navbar-right">
    <span class="navbar-status">{st.session_state.navbar_status}</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ── TWO-COLUMN LAYOUT ─────────────────────────────────────────────────────────
left_col, right_col = st.columns([2.4, 1], gap="large")

with left_col:
    # Hero
    st.markdown("""
    <div style="padding: 32px 0 0 0;">
      <div class="hero-title">Make any model<br><span class="green">explain itself.</span></div>
      <div class="hero-subtitle">
        Upload your trained ML model and the dataset it was trained on.
        XAI-Agent uses Hermes Agent to run real SHAP &amp; LIME analysis and produces
        a plain-English report any stakeholder can read.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # File uploaders
    st.markdown("**Drop model + dataset, or click to browse**", unsafe_allow_html=False)
    st.caption("Model: .pkl .joblib · Dataset: .csv (required)")

    model_file   = st.file_uploader("Upload model",   type=["pkl","joblib"], label_visibility="collapsed", key="model_up")
    dataset_file = st.file_uploader("Upload dataset", type=["csv"],          label_visibility="collapsed", key="data_up")

    # File chips
    if model_file or dataset_file:
        chips_html = '<div class="file-chips">'
        if model_file:
            size_kb = len(model_file.getvalue()) // 1024
            chips_html += f'''
            <div class="file-chip">
              <div class="chip-icon model">📦</div>
              <div>
                <div class="chip-name">{model_file.name}</div>
                <div class="chip-meta">scikit-learn / pickle · {size_kb} KB</div>
              </div>
            </div>'''
        if dataset_file:
            size_kb = len(dataset_file.getvalue()) // 1024
            chips_html += f'''
            <div class="file-chip">
              <div class="chip-icon data">📊</div>
              <div>
                <div class="chip-name">{dataset_file.name}</div>
                <div class="chip-meta">Dataset · {size_kb} KB</div>
              </div>
            </div>'''
        chips_html += '</div>'
        st.markdown(chips_html, unsafe_allow_html=True)
        st.session_state.navbar_status = f"Model: {model_file.name}" if model_file else "Dataset loaded"

    # ── AGENT STEP TRACKER (shown during/after run) ──────────────────────────
    if any(s != "pending" for s in st.session_state.step_status):
        step_names = [s[0] for s in HermesXAIAgent.STEPS]
        step_descs = [s[1] for s in HermesXAIAgent.STEPS]
        steps_html = '<div class="agent-steps"><div class="agent-steps-title">🤖 Hermes Agent — live progress</div>'
        for i, (name, desc) in enumerate(zip(step_names, step_descs)):
            status = st.session_state.step_status[i]
            if status == "done":
                dot_cls, symbol = "done", "✓"
            elif status == "running":
                dot_cls, symbol = "running", "⟳"
            else:
                dot_cls, symbol = "pending", str(i+1)
            steps_html += f'''
            <div class="step-item">
              <div class="step-dot {dot_cls}">{symbol}</div>
              <div class="step-text">
                <div class="step-name">{name}</div>
                <div class="step-desc">{desc}</div>
              </div>
            </div>'''
        steps_html += '</div>'
        st.markdown(steps_html, unsafe_allow_html=True)

    # ── REPORT OUTPUT ─────────────────────────────────────────────────────────
    if st.session_state.report_done:
        r = st.session_state.agent_results

        # Executive summary
        imp_df = r["imp_df"]
        top_feat = imp_df.iloc[0]["feature"]
        model_type = r["model_type"]
        X = r["X"]
        feat_cols = r["feat_cols"]
        bias_flag = r["bias_flag"]
        bias_msg  = r["bias_msg"]

        st.markdown(f"""
        <div class="exec-summary">
          <div class="exec-summary-title">📋 Executive Summary</div>
          <div class="exec-summary-text">
            This <strong>{model_type}</strong> model was analyzed across
            <strong>{len(X)} samples</strong> and <strong>{len(feat_cols)} features</strong>.
            The single most influential predictor is <strong>'{top_feat}'</strong>.
            {"⚠️ Bias review is recommended." if bias_flag else "✅ No obvious bias was detected."}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # SHAP Chart
        st.markdown('<div class="report-section"><div class="report-section-title">📊 Feature Importance (SHAP)</div>', unsafe_allow_html=True)
        st.image(r["shap_plot"], use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Top features bar
        st.markdown('<div class="report-section"><div class="report-section-title">🏆 Top 5 Driving Features</div>', unsafe_allow_html=True)
        top5 = imp_df.head(5)
        max_val = top5["importance"].max()
        bars_html = ""
        for _, row in top5.iterrows():
            pct = int((row["importance"] / max_val) * 100)
            color = "#1a56db" if row["direction"] == "↑ positive" else "#059669"
            bars_html += f"""
            <div class="feat-row">
              <div class="feat-label">
                <span>{row['feature']}</span>
                <span style="color:#64748B">{row['importance']:.4f} · {row['direction']}</span>
              </div>
              <div class="feat-bar-bg">
                <div class="feat-bar-fill" style="width:{pct}%;background:{color}"></div>
              </div>
            </div>"""
        st.markdown(bars_html + '</div>', unsafe_allow_html=True)

        # LIME explanations
        st.markdown('<div class="report-section"><div class="report-section-title">🔍 Sample Predictions Explained</div>', unsafe_allow_html=True)
        for i, res in enumerate(r["lime_results"], 1):
            st.markdown(f"**Sample {i} — Row {res['row']}**")
            st.markdown(res["plain"])
            for feat, w in res["reasons"][:3]:
                arrow = "🟢" if w > 0 else "🔴"
                st.markdown(f"  {arrow} `{feat}` → `{w:+.3f}`")
            st.divider()
        st.markdown('</div>', unsafe_allow_html=True)

        # Bias check
        st.markdown('<div class="report-section"><div class="report-section-title">⚖️ Bias & Fairness Check</div>', unsafe_allow_html=True)
        badge_cls = "bias-warn" if bias_flag else "bias-ok"
        badge_txt = "⚠️ Review recommended" if bias_flag else "✅ No obvious bias"
        st.markdown(f'<span class="{badge_cls}">{badge_txt}</span>', unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:13px;color:#475569;margin-top:10px'>{bias_msg}</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Downloads
        st.markdown("### 📥 Download Report")
        report_md = r["report_md"]
        b64_md = base64.b64encode(report_md.encode()).decode()
        shap_b64 = base64.b64encode(r["shap_plot"].getvalue()).decode()

        st.markdown(f"""
        <a href="data:text/markdown;base64,{b64_md}" download="xai_report.md" class="dl-btn">
          📄 Download Report (.md)
        </a>
        <a href="data:image/png;base64,{shap_b64}" download="shap_chart.png" class="dl-btn outline">
          📊 Download SHAP Chart
        </a>
        """, unsafe_allow_html=True)

        # Footer
        method = r["explainer_method"]
        st.markdown(f"""
        <div class="footer-note">
          XAI-Agent ran a full structural inspection using <strong>{method}</strong>.
          Powered by <strong>Hermes Agent</strong> multi-step planning pipeline.
          For production audits, pair with a Python backend running SHAP/LIME on your full dataset.
        </div>
        """, unsafe_allow_html=True)


# ── RIGHT PANEL ───────────────────────────────────────────────────────────────
with right_col:
    st.markdown("""
    <div style="padding-top: 32px;">
      <div class="settings-title">Analysis settings</div>
    </div>
    """, unsafe_allow_html=True)

    # Task type
    st.markdown('<div class="settings-label">Task Type</div>', unsafe_allow_html=True)
    task_col1, task_col2, task_col3 = st.columns(3)
    with task_col1:
        if st.button("Classification", key="cls_btn",
                     type="primary" if st.session_state.task_type=="Classification" else "secondary",
                     use_container_width=True):
            st.session_state.task_type = "Classification"
    with task_col2:
        if st.button("Regression", key="reg_btn",
                     type="primary" if st.session_state.task_type=="Regression" else "secondary",
                     use_container_width=True):
            st.session_state.task_type = "Regression"
    with task_col3:
        if st.button("Auto", key="auto_btn",
                     type="primary" if st.session_state.task_type=="Auto" else "secondary",
                     use_container_width=True):
            st.session_state.task_type = "Auto"

    st.markdown("<br>", unsafe_allow_html=True)

    # Target column
    st.markdown('<div class="settings-label">Target Column (optional)</div>', unsafe_allow_html=True)
    target_col_input = st.text_input(
        "Target column", value="target",
        placeholder="e.g. churned, price, label",
        label_visibility="collapsed"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # RUN BUTTON
    run_disabled = not (model_file and dataset_file)
    run_clicked  = st.button(
        "✦  Run explainability analysis",
        disabled=run_disabled,
        use_container_width=True,
        type="primary",
        key="run_btn"
    )

    st.markdown("""
    <div class="run-btn-hint">
      ℹ️ Files stay in your browser — only column names and a small sample are sent for analysis.
    </div>
    """, unsafe_allow_html=True)

    # Hermes Agent branding
    st.markdown("""
    <div style="margin-top: 32px; padding: 14px; background: #F8FAFC;
                border-radius: 10px; border: 1px solid #E2E8F0;">
      <div style="font-size: 12px; font-weight: 600; color: #0F172A; margin-bottom: 6px;">
        🤖 Powered by Hermes Agent
      </div>
      <div style="font-size: 11px; color: #64748B; line-height: 1.6;">
        Multi-step autonomous agent pipeline:<br>
        Inspect → SHAP → LIME → Bias → Report
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── RUN AGENT ─────────────────────────────────────────────────────────────────
if run_clicked and model_file and dataset_file:
    st.session_state.report_done = False
    st.session_state.step_status = ["pending"] * 5

    agent = HermesXAIAgent()
    progress = st.progress(0, text="🤖 Hermes Agent starting...")

    try:
        # Step 1
        st.session_state.step_status[0] = "running"
        progress.progress(10, text="Step 1/5 — Inspecting model & dataset...")
        model_type, n_feats, n_rows, task = agent.tool_file_reader(
            model_file.getvalue(),
            dataset_file.getvalue(),
            target_col_input or "target"
        )
        st.session_state.step_status[0] = "done"

        # Step 2
        st.session_state.step_status[1] = "running"
        progress.progress(35, text="Step 2/5 — Running SHAP analysis...")
        imp_df, shap_plot = agent.tool_shap_analyzer()
        st.session_state.step_status[1] = "done"

        # Step 3
        st.session_state.step_status[2] = "running"
        progress.progress(58, text="Step 3/5 — Running LIME local explanations...")
        lime_results = agent.tool_lime_explainer()
        st.session_state.step_status[2] = "done"

        # Step 4
        st.session_state.step_status[3] = "running"
        progress.progress(78, text="Step 4/5 — Checking for bias...")
        bias_flag, bias_msg = agent.tool_bias_checker()
        st.session_state.step_status[3] = "done"

        # Step 5
        st.session_state.step_status[4] = "running"
        progress.progress(92, text="Step 5/5 — Writing report...")
        report_md = agent.tool_report_writer()
        st.session_state.step_status[4] = "done"

        progress.progress(100, text="✅ Analysis complete!")
        time.sleep(0.5)
        progress.empty()

        # Save to session state
        st.session_state.agent_results = agent.results
        st.session_state.agent_results["shap_plot"] = shap_plot
        st.session_state.agent_results["report_md"] = report_md
        st.session_state.report_done = True
        st.session_state.navbar_status = f"✅ Analysis complete — {n_rows} rows, {n_feats} features"
        st.rerun()

    except Exception as e:
        progress.empty()
        st.error(f"❌ Agent error: {str(e)}")
        st.info("Make sure your model is a valid .pkl file and your dataset is a .csv with the correct target column name.")