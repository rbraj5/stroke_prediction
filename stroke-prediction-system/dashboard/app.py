"""Streamlit dashboard that doubles as a prediction UI and an interview walkthrough surface."""
from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.getenv("STROKE_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Stroke Screening Portfolio", page_icon=":hospital:", layout="wide")

st.markdown(
    """
    <style>
    .summary-card {
        padding: 1rem 1.25rem;
        border-radius: 14px;
        background: linear-gradient(135deg, #122033 0%, #1f3959 100%);
        color: #f4f7fb;
        margin-bottom: 1rem;
    }
    .decision-card {
        padding: 1rem 1.25rem;
        border-radius: 14px;
        background: #f5f8fc;
        border: 1px solid #dfe8f3;
        color: #203247;
    }
    .risk-pill {
        display: inline-block;
        padding: 0.3rem 0.7rem;
        border-radius: 999px;
        font-weight: 600;
        margin-top: 0.25rem;
    }
    .risk-low {background:#e7f6eb;color:#146a32;}
    .risk-moderate {background:#fff4de;color:#8b5a00;}
    .risk-high {background:#fde9e8;color:#9f1c1c;}
    </style>
    """,
    unsafe_allow_html=True,
)


def api_get(path: str):
    return requests.get(f"{API_URL}{path}", timeout=5)


@st.cache_data(show_spinner=False)
def get_model_metadata():
    try:
        response = api_get("/model/metadata")
        if response.status_code == 200:
            return response.json()
    except requests.RequestException:
        return None
    return None


@st.cache_data(show_spinner=False)
def get_health():
    try:
        response = api_get("/health")
        if response.status_code == 200:
            return response.json()
    except requests.RequestException:
        return None
    return None



def predict(patient_data: dict):
    try:
        response = requests.post(f"{API_URL}/predict", json=patient_data, timeout=10)
        if response.status_code == 200:
            return response.json()
        return {"error": response.json().get("detail", "Prediction failed")}
    except requests.RequestException as exc:
        return {"error": str(exc)}



def feature_label(feature: str, metadata: dict) -> str:
    labels = metadata["feature_decisions"].get("display_feature_labels", {})
    return labels.get(feature, feature.replace("_", " ").title())



def build_confusion_matrix_figure(confusion_matrix: list[list[int]]) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=confusion_matrix,
            x=["Predicted negative", "Predicted positive"],
            y=["Actual negative", "Actual positive"],
            colorscale="Blues",
            text=confusion_matrix,
            texttemplate="%{text}",
        )
    )
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
    return fig



def build_model_comparison_figure(candidate_models: list[dict]) -> go.Figure:
    comparison_df = pd.DataFrame(
        [
            {
                "Model": model["label"],
                "Recall": model["metrics"]["recall"],
                "Precision": model["metrics"]["precision"],
                "F1": model["metrics"]["f1_score"],
                "ROC-AUC": model["metrics"]["roc_auc"],
            }
            for model in candidate_models
        ]
    )
    fig = px.bar(
        comparison_df,
        x="Model",
        y=["Recall", "Precision", "F1", "ROC-AUC"],
        barmode="group",
        height=450,
        title="Model comparison across screening-relevant metrics",
    )
    return fig



def build_explanation_chart(explanation_summary: list[dict], metadata: dict):
    if not explanation_summary:
        return None
    explanation_df = pd.DataFrame(
        [
            {
                "Feature": feature_label(item["feature"], metadata),
                "Contribution": item["contribution"],
                "Direction": item["direction"],
            }
            for item in explanation_summary
        ]
    )
    fig = px.bar(
        explanation_df,
        x="Contribution",
        y="Feature",
        color="Direction",
        orientation="h",
        height=350,
        title="Top factors in the current prediction summary",
    )
    return fig



def build_workflow_map_figure() -> go.Figure:
    labels = [
        "Raw dataset",
        "Cleaning & BMI median imputation",
        "Feature selection",
        "Model comparison",
        "Threshold tuning",
        "Saved artifacts",
        "FastAPI",
        "Dashboard",
    ]
    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                pad=18,
                thickness=18,
                line=dict(color="#0f1720", width=1),
                color=["#203247", "#28527a", "#3b82a0", "#4f46e5", "#f59e0b", "#10b981", "#ef4444", "#7c3aed"],
                label=labels,
            ),
            link=dict(
                source=[0, 1, 2, 3, 4, 5, 6],
                target=[1, 2, 3, 4, 5, 6, 7],
                value=[10, 10, 10, 10, 10, 10, 10],
                color=["rgba(59,130,160,0.35)"] * 7,
            ),
        )
    )
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10), title="Workflow map")
    return fig



def build_architecture_figure() -> go.Figure:
    nodes = [
        (0.10, 0.50, "Dataset + Notebook\nResearch baseline", "#203247"),
        (0.32, 0.50, "Training Pipeline\npreprocessing.py + train.py", "#28527a"),
        (0.55, 0.72, "Model Artifacts\npkl + metadata json", "#10b981"),
        (0.55, 0.28, "FastAPI Service\npredict + metadata", "#ef4444"),
        (0.80, 0.50, "Streamlit App\ninterviewer-facing UI", "#7c3aed"),
    ]
    fig = go.Figure()
    for x0, y0, x1, y1 in [
        (0.18, 0.50, 0.25, 0.50),
        (0.39, 0.56, 0.48, 0.67),
        (0.39, 0.44, 0.48, 0.33),
        (0.62, 0.28, 0.72, 0.45),
        (0.62, 0.72, 0.72, 0.55),
    ]:
        fig.add_shape(type="line", x0=x0, y0=y0, x1=x1, y1=y1, line=dict(color="#94a3b8", width=3))
    for x, y, label, color in nodes:
        fig.add_shape(type="rect", x0=x - 0.10, y0=y - 0.08, x1=x + 0.10, y1=y + 0.08, line=dict(color=color, width=2), fillcolor="rgba(255,255,255,0.96)")
        fig.add_annotation(x=x, y=y, text=label, showarrow=False, font=dict(size=13, color="#0f1720"))
    fig.update_layout(
        title="Architecture sketch",
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1]),
        height=360,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig



def risk_class_name(risk_level: str) -> str:
    mapping = {"Low": "risk-low", "Moderate": "risk-moderate", "High": "risk-high"}
    return mapping.get(risk_level, "risk-low")



def render_overview(metadata: dict):
    active_model = metadata["active_model"]
    candidate_models = metadata["candidate_models"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Active Model", active_model["label"])
    col2.metric("Decision Threshold", f"{active_model['threshold']:.2f}")
    col3.metric("Selected Features", metadata["feature_decisions"]["selected_feature_count"])
    col4.metric("Models Evaluated", len(candidate_models))

    st.markdown(
        f"""
        <div class="summary-card">
            <h3 style="margin-top:0;">Use Case</h3>
            <p style="margin-bottom:0;">{active_model['use_case']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.25, 1])
    with left:
        st.markdown(
            f"""
            <div class="decision-card">
                <h4 style="margin-top:0;">Decision Summary</h4>
                <p><strong>Why this version:</strong> {active_model['why_this_version']}</p>
                <p><strong>Primary objective:</strong> {active_model['primary_objective']}</p>
                <p><strong>Threshold rationale:</strong> {active_model['threshold_rationale']}</p>
                <p><strong>Model version:</strong> {active_model['version']}</p>
                <p style="margin-bottom:0;"><strong>Training date:</strong> {active_model['training_date']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        metrics = active_model["metrics"]
        st.subheader("Evaluation Snapshot")
        st.metric("Recall", f"{metrics['recall']:.2%}")
        st.metric("Precision", f"{metrics['precision']:.2%}")
        st.metric("F1", f"{metrics['f1_score']:.2%}")
        st.metric("ROC-AUC", f"{metrics['roc_auc']:.2%}")

    st.plotly_chart(build_workflow_map_figure(), use_container_width=True)



def render_predict_tab(metadata: dict):
    st.subheader("Predict")
    st.caption("This form mirrors the stable API contract. It shows only the inputs used by the current prediction workflow.")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=45)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda value: "Yes" if value else "No")
    with col2:
        heart_disease = st.selectbox("Heart disease", [0, 1], format_func=lambda value: "Yes" if value else "No")
        avg_glucose_level = st.number_input("Average glucose level (mg/dL)", min_value=0.0, max_value=300.0, value=100.0)
        bmi = st.number_input("BMI", min_value=10.0, max_value=100.0, value=25.0)
    with col3:
        smoking_status = st.selectbox("Smoking status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

    if st.button("Run screening prediction", type="primary", use_container_width=True):
        patient_data = {
            "age": age,
            "gender": gender,
            "hypertension": hypertension,
            "heart_disease": heart_disease,
            "avg_glucose_level": avg_glucose_level,
            "bmi": bmi,
            "smoking_status": smoking_status,
        }
        result = predict(patient_data)
        if "error" in result:
            st.error(result["error"])
            return

        left, middle, right = st.columns(3)
        left.metric("Current Active Model", result["active_model"])
        middle.metric("Probability", f"{result['probability']:.1%}")
        right.metric("Decision Threshold", f"{result['threshold']:.2f}")

        st.markdown(
            f'<span class="risk-pill {risk_class_name(result["risk_level"])}">{result["risk_level"]} screening risk</span>',
            unsafe_allow_html=True,
        )
        st.write(result["note"])

        explanation_chart = build_explanation_chart(result.get("explanation_summary", []), metadata)
        if explanation_chart:
            st.plotly_chart(explanation_chart, use_container_width=True)

        if result.get("explanation_summary"):
            st.markdown("**Plain-language factor summary**")
            for item in result["explanation_summary"]:
                st.write(f"- {feature_label(item['feature'], metadata)}: {item['direction']}")



def render_model_comparison_tab(metadata: dict):
    st.subheader("Model Comparison")
    st.caption("This section makes the comparison-and-selection workflow visible without opening the training code.")
    st.plotly_chart(build_model_comparison_figure(metadata["candidate_models"]), use_container_width=True)

    comparison_df = pd.DataFrame(
        [
            {
                "Model": model["label"],
                "Status": model["status"],
                "Recall": f"{model['metrics']['recall']:.2%}",
                "Precision": f"{model['metrics']['precision']:.2%}",
                "F1": f"{model['metrics']['f1_score']:.2%}",
                "ROC-AUC": f"{model['metrics']['roc_auc']:.2%}",
                "Selection rationale": model["selection_rationale"],
                "Why not active": model["rejection_rationale"],
            }
            for model in metadata["candidate_models"]
        ]
    )
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)



def render_feature_tab(metadata: dict):
    st.subheader("Feature & Data Decisions")
    feature_decisions = metadata["feature_decisions"]
    selected_features = feature_decisions["selected_features"]

    left, right = st.columns(2)
    with left:
        st.markdown("**Selected Features**")
        for feature in selected_features:
            st.write(f"- {feature_label(feature, metadata)}")
    with right:
        st.markdown("**Excluded Features**")
        for feature, rationale in feature_decisions["excluded_features"].items():
            st.write(f"- {feature}: {rationale}")

    st.markdown("**Preprocessing Decisions**")
    for title, rationale in feature_decisions["preprocessing_decisions"].items():
        label = title.replace("_", " ").title()
        st.write(f"- {label}: {rationale}")



def render_performance_tab(metadata: dict):
    st.subheader("Performance & Threshold")
    active_model = metadata["active_model"]
    metrics = active_model["metrics"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Recall", f"{metrics['recall']:.2%}")
    col2.metric("Precision", f"{metrics['precision']:.2%}")
    col3.metric("F1", f"{metrics['f1_score']:.2%}")
    col4.metric("ROC-AUC", f"{metrics['roc_auc']:.2%}")

    left, right = st.columns([1.1, 0.9])
    with left:
        st.plotly_chart(build_confusion_matrix_figure(active_model["confusion_matrix"]), use_container_width=True)
    with right:
        st.markdown("**Threshold Rationale**")
        st.write(active_model["threshold_rationale"])
        st.markdown("**Healthcare Tradeoff**")
        st.write(
            "The current system treats false negatives as the more serious mistake. "
            "It accepts more false positives if that helps surface more at-risk patients for follow-up screening."
        )



def render_workflow_tab(metadata: dict):
    st.subheader("Workflow & Engineering")
    workflow = metadata["workflow_summary"]

    left, right = st.columns(2)
    with left:
        st.plotly_chart(build_workflow_map_figure(), use_container_width=True)
    with right:
        st.plotly_chart(build_architecture_figure(), use_container_width=True)

    details_left, details_right = st.columns(2)
    with details_left:
        st.markdown("**Workflow Steps**")
        for step in workflow["steps"]:
            st.write(f"- {step}")
    with details_right:
        st.markdown("**Engineering Signals**")
        for signal in workflow["engineering_signals"]:
            st.write(f"- {signal}")



def render_limitations_tab(metadata: dict):
    st.subheader("Limitations & Next Steps")
    left, right = st.columns(2)
    with left:
        st.markdown("**Current Limitations**")
        for item in metadata["limitations"]:
            st.write(f"- {item}")
    with right:
        st.markdown("**Next Steps**")
        for item in metadata["next_steps"]:
            st.write(f"- {item}")



def main():
    st.title("Stroke Risk Screening Portfolio")
    st.markdown("A research-to-product walkthrough of a stroke risk screening workflow.")
    st.caption(f"Dashboard API source: {API_URL}")

    health = get_health()
    metadata = get_model_metadata()

    if not health or not metadata or not health.get("model_loaded"):
        st.error("API or model metadata is unavailable. Start the FastAPI server and ensure the latest artifacts exist in models/.")
        st.code("python -m uvicorn api.main:app --reload", language="bash")
        return

    overview, predict_tab, comparison_tab, feature_tab, performance_tab, workflow_tab, limitations_tab = st.tabs(
        [
            "Overview",
            "Predict",
            "Model Comparison",
            "Feature & Data Decisions",
            "Performance & Threshold",
            "Workflow & Engineering",
            "Limitations & Next Steps",
        ]
    )

    with overview:
        render_overview(metadata)
    with predict_tab:
        render_predict_tab(metadata)
    with comparison_tab:
        render_model_comparison_tab(metadata)
    with feature_tab:
        render_feature_tab(metadata)
    with performance_tab:
        render_performance_tab(metadata)
    with workflow_tab:
        render_workflow_tab(metadata)
    with limitations_tab:
        render_limitations_tab(metadata)


if __name__ == "__main__":
    main()
