import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

# ── Page Configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bangalore Traffic Analytics | Descriptive & Predictive Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom White & Green Theme CSS ─────────────────────────────────────────
st.markdown("""
<style>
    /* Main App Background & Typography */
    .stApp {
        background-color: #FFFFFF;
        color: #143321;
    }
    [data-testid="stSidebar"] {
        background-color: #F2F9F5;
        border-right: 1px solid #C6E6D3;
    }
    
    /* Top Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #14532D 0%, #1B7A43 60%, #229954 100%);
        color: #FFFFFF;
        padding: 1.6rem 2rem;
        border-radius: 14px;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 14px rgba(20, 83, 45, 0.14);
    }
    .hero-banner h1 {
        color: #FFFFFF !important;
        margin: 0 0 0.35rem 0;
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.3px;
    }
    .hero-banner p {
        color: #E2F5EA;
        margin: 0;
        font-size: 0.98rem;
    }
    .badge-pill {
        display: inline-block;
        background-color: rgba(255, 255, 255, 0.18);
        color: #FFFFFF;
        padding: 0.22rem 0.75rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }

    /* Custom KPI Metric Cards */
    .kpi-card {
        background-color: #F4FAF6;
        border: 1px solid #CBEAD7;
        border-left: 5px solid #1B7A43;
        border-radius: 12px;
        padding: 1.05rem 1.2rem;
        box-shadow: 0 2px 6px rgba(27, 122, 67, 0.06);
        margin-bottom: 0.5rem;
    }
    .kpi-label {
        font-size: 0.80rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #2E6F48;
        margin-bottom: 0.25rem;
    }
    .kpi-value {
        font-size: 1.65rem;
        font-weight: 800;
        color: #14532D;
        line-height: 1.2;
    }
    .kpi-sub {
        font-size: 0.78rem;
        color: #3D7A57;
        margin-top: 0.25rem;
    }

    /* Section Headers */
    .section-header {
        color: #14532D;
        font-weight: 800;
        font-size: 1.28rem;
        border-bottom: 2px solid #D1EADF;
        padding-bottom: 0.4rem;
        margin-top: 0.8rem;
        margin-bottom: 1rem;
    }

    /* Prediction Result Cards */
    .pred-card-high {
        background: linear-gradient(135deg, #FFF5F5 0%, #FEE2E2 100%);
        border: 2px solid #DC2626;
        border-left: 8px solid #B91C1C;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        color: #7F1D1D;
    }
    .pred-card-low {
        background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
        border: 2px solid #16A34A;
        border-left: 8px solid #14532D;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        color: #14532D;
    }
    .info-box-green {
        background-color: #F2F9F5;
        border: 1px solid #B7E4C7;
        border-left: 4px solid #1B7A43;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        color: #143321;
        margin-bottom: 1rem;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #F2F9F5;
        padding: 6px 8px;
        border-radius: 12px;
        border: 1px solid #CBEAD7;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 18px;
        font-weight: 700;
        color: #1B7A43;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1B7A43 !important;
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Load & Prepare Dataset ─────────────────────────────────────────────────
@st.cache_data
def load_and_prepare_data():
    df_raw = pd.read_csv("Banglore_traffic_Dataset.csv")
    df = df_raw.copy()
    df["Parsed_Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Parsed_Date"].dt.year
    df["Month"] = df["Parsed_Date"].dt.month
    df["DayOfWeek"] = df["Parsed_Date"].dt.day_name()
    df["IsWeekend"] = (df["Parsed_Date"].dt.dayofweek >= 5).astype(int)
    df["high_congestion"] = (df["Congestion Level"] >= 80.0).astype(int)
    df["Congestion_Class_Label"] = np.where(
        df["high_congestion"] == 1, "High Congestion (≥80%)", "Moderate/Low (<80%)"
    )

    # Create cleaned dataset matching Deliverable 3 notebook
    drop_cols = [
        "Congestion Level",
        "Environmental Impact",
        "Road Capacity Utilization",
        "Travel Time Index",
        "Public Transport Usage",
        "Traffic Signal Compliance",
        "Parking Usage",
        "Date",
        "Parsed_Date",
        "Year",
        "Congestion_Class_Label",
    ]
    df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()
    return df, df_clean

df, df_clean = load_and_prepare_data()

# ── Train & Cache All 5 Benchmark Models + Baseline ────────────────────────
NUMERIC_FEATURES = [
    "Traffic Volume",
    "Average Speed",
    "Incident Reports",
    "Pedestrian and Cyclist Count",
    "Month",
    "IsWeekend",
]
CATEGORICAL_FEATURES = [
    "Area Name",
    "Road/Intersection Name",
    "Weather Conditions",
    "Roadwork and Construction Activity",
    "DayOfWeek",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

@st.cache_resource
def train_all_models(_df_clean, _df_full):
    X = _df_clean[ALL_FEATURES]
    y = _df_clean["high_congestion"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )

    models_spec = {
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        "Support Vector Machine (RBF)": SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=11, weights="distance"),
        "Gaussian Naive Bayes": GaussianNB(),
        "Majority Baseline (Always High)": DummyClassifier(strategy="most_frequent"),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipelines = {}
    metrics_rows = []
    roc_data = {}
    cm_data = {}

    for name, clf in models_spec.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
        cv_acc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        cv_rec = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="recall", n_jobs=-1)
        pipe.fit(X_train, y_train)
        pipelines[name] = pipe

        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr, auc)
        cm_data[name] = confusion_matrix(y_test, y_pred)

        metrics_rows.append({
            "Model": name,
            "CV Accuracy (%)": round(cv_acc.mean() * 100, 2),
            "CV Recall (%)": round(cv_rec.mean() * 100, 2),
            "Test Accuracy (%)": round(acc * 100, 2),
            "Gain vs Baseline (%)": round((acc - 0.621924) * 100, 2),
            "Test Precision (%)": round(prec * 100, 2),
            "Test Recall (%)": round(rec * 100, 2),
            "Test F1-Score": round(f1, 4),
            "Test ROC-AUC": round(auc, 4),
        })

    # Extract Random Forest Feature Importances
    rf_pipe = pipelines["Random Forest"]
    ohe_names = list(
        rf_pipe.named_steps["preprocessor"]
        .named_transformers_["cat"]
        .get_feature_names_out(CATEGORICAL_FEATURES)
    )
    feat_names = NUMERIC_FEATURES + ohe_names
    feat_imp_df = (
        pd.DataFrame({"Feature": feat_names, "Importance": rf_pipe.named_steps["classifier"].feature_importances_})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    # Diagnostic 1: Ablation without Traffic Volume & Average Speed
    num_ablation = ["Incident Reports", "Pedestrian and Cyclist Count", "Month", "IsWeekend"]
    prep_ablation = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_ablation),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )
    ablation_rows = []
    for m_name in ["Random Forest", "Logistic Regression", "Support Vector Machine (RBF)"]:
        p_ab = Pipeline([("preprocessor", prep_ablation), ("classifier", models_spec[m_name])])
        p_ab.fit(X_train[num_ablation + CATEGORICAL_FEATURES], y_train)
        yp_ab = p_ab.predict(X_test[num_ablation + CATEGORICAL_FEATURES])
        ypr_ab = p_ab.predict_proba(X_test[num_ablation + CATEGORICAL_FEATURES])[:, 1]
        ablation_rows.append({
            "Model": m_name,
            "Full 11 Features Acc (%)": round(
                next(r["Test Accuracy (%)"] for r in metrics_rows if r["Model"] == m_name), 2
            ),
            "Without Speed & Volume Acc (%)": round(accuracy_score(y_test, yp_ab) * 100, 2),
            "Without Speed & Volume Recall (%)": round(recall_score(y_test, yp_ab) * 100, 2),
            "Without Speed & Volume ROC-AUC": round(roc_auc_score(y_test, ypr_ab), 4),
        })

    # Diagnostic 2: Chronological Split (2022-2023 Train -> 2024 Test)
    train_mask = _df_full["Year"] <= 2023
    test_mask = _df_full["Year"] == 2024
    X_tr_c, y_tr_c = X[train_mask], y[train_mask]
    X_te_c, y_te_c = X[test_mask], y[test_mask]
    chrono_rows = []
    for m_name in ["Random Forest", "Logistic Regression", "Support Vector Machine (RBF)"]:
        p_ch = Pipeline([("preprocessor", preprocessor), ("classifier", models_spec[m_name])])
        p_ch.fit(X_tr_c, y_tr_c)
        yp_ch = p_ch.predict(X_te_c)
        ypr_ch = p_ch.predict_proba(X_te_c)[:, 1]
        chrono_rows.append({
            "Model": m_name,
            "Stratified 80/20 Acc (%)": round(
                next(r["Test Accuracy (%)"] for r in metrics_rows if r["Model"] == m_name), 2
            ),
            "Chronological (2024 Test) Acc (%)": round(accuracy_score(y_te_c, yp_ch) * 100, 2),
            "Chronological (2024 Test) Recall (%)": round(recall_score(y_te_c, yp_ch) * 100, 2),
            "Chronological (2024 Test) ROC-AUC": round(roc_auc_score(y_te_c, ypr_ch), 4),
        })

    return {
        "pipelines": pipelines,
        "metrics_df": pd.DataFrame(metrics_rows),
        "roc_data": roc_data,
        "cm_data": cm_data,
        "feat_imp_df": feat_imp_df,
        "ablation_df": pd.DataFrame(ablation_rows),
        "chrono_df": pd.DataFrame(chrono_rows),
        "test_counts": {"total": len(y_test), "high": int((y_test == 1).sum()), "low": int((y_test == 0).sum())},
    }

artifacts = train_all_models(df_clean, df)

# ── Sidebar Controls & Global Filters ──────────────────────────────────────
st.sidebar.markdown("## 🌿 Group 5.1 Traffic Analytics")
st.sidebar.caption("Prepared by **John Loyd Arcilla** & **Sean Mikheal Fulgencio**")
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Descriptive Data Filters")

min_date = df["Parsed_Date"].min().date()
max_date = df["Parsed_Date"].max().date()

date_selection = st.sidebar.date_input(
    "📅 Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)
if isinstance(date_selection, tuple) and len(date_selection) == 2:
    start_date, end_date = pd.to_datetime(date_selection[0]), pd.to_datetime(date_selection[1])
else:
    start_date, end_date = pd.to_datetime(min_date), pd.to_datetime(max_date)

all_areas = ["All Sectors"] + sorted(df["Area Name"].unique().tolist())
selected_area = st.sidebar.selectbox("🏙️ Urban Sector (Area Name)", all_areas)

if selected_area != "All Sectors":
    available_roads = ["All Intersections"] + sorted(
        df.loc[df["Area Name"] == selected_area, "Road/Intersection Name"].unique().tolist()
    )
else:
    available_roads = ["All Intersections"] + sorted(df["Road/Intersection Name"].unique().tolist())
selected_road = st.sidebar.selectbox("🛣️ Road / Intersection", available_roads)

all_weather = ["All Weather"] + sorted(df["Weather Conditions"].unique().tolist())
selected_weather = st.sidebar.selectbox("🌦️ Weather Condition", all_weather)

all_roadwork = ["All"] + sorted(df["Roadwork and Construction Activity"].unique().tolist())
selected_roadwork = st.sidebar.selectbox("🚧 Roadwork / Construction", all_roadwork)

# Apply Filters
mask = (df["Parsed_Date"] >= start_date) & (df["Parsed_Date"] <= end_date)
if selected_area != "All Sectors":
    mask &= df["Area Name"] == selected_area
if selected_road != "All Intersections":
    mask &= df["Road/Intersection Name"] == selected_road
if selected_weather != "All Weather":
    mask &= df["Weather Conditions"] == selected_weather
if selected_roadwork != "All":
    mask &= df["Roadwork and Construction Activity"] == selected_roadwork

filtered_df = df.loc[mask].copy()

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Filtered Records:** `{len(filtered_df):,}` / `{len(df):,}` rows\n\n"
    f"**Target Definition:** High Congestion = `Congestion Level ≥ 80%`"
)

# ── Top Hero Banner ────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div>
        <span class="badge-pill">Group 5.1 · Arcilla & Fulgencio</span>
        <span class="badge-pill">Bangalore Urban Traffic Dataset (8,936 Observations)</span>
        <span class="badge-pill">Supervised Binary Classification</span>
    </div>
    <h1>🌿 Bangalore Urban Traffic Congestion Analytics Dashboard</h1>
    <p>Proactive Decision-Support System for Municipal Traffic Authorities, Urban Planners, and Traffic Engineers — Combining Historical Exploratory Analysis with Live Machine Learning Classification.</p>
</div>
""", unsafe_allow_html=True)

# ── Navigation Tabs ────────────────────────────────────────────────────────
tab_desc, tab_pred, tab_eval, tab_found = st.tabs([
    "📊 1. Descriptive Analytics (EDA)",
    "🤖 2. Predictive Congestion Simulator (Live ML)",
    "📈 3. Model Benchmarking & Diagnostics",
    "📋 4. Project Foundation & Dataset Explorer"
])

# ===========================================================================
# TAB 1: DESCRIPTIVE ANALYTICS (EDA)
# ===========================================================================
with tab_desc:
    if filtered_df.empty:
        st.warning("No observations match the selected sidebar filters. Please broaden your filter criteria.")
    else:
        # KPI Cards Row
        k1, k2, k3, k4, k5 = st.columns(5)
        obs_count = len(filtered_df)
        high_cong_rate = filtered_df["high_congestion"].mean() * 100
        mean_volume = filtered_df["Traffic Volume"].mean()
        mean_speed = filtered_df["Average Speed"].mean()
        total_inc = filtered_df["Incident Reports"].sum()

        with k1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Observations</div>
                <div class="kpi-value">{obs_count:,}</div>
                <div class="kpi-sub">{filtered_df['Parsed_Date'].nunique()} distinct dates</div>
            </div>""", unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">High Congestion Rate</div>
                <div class="kpi-value">{high_cong_rate:.1f}%</div>
                <div class="kpi-sub">{int(filtered_df['high_congestion'].sum()):,} High (≥80%) days</div>
            </div>""", unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Mean Traffic Volume</div>
                <div class="kpi-value">{mean_volume:,.0f}</div>
                <div class="kpi-sub">Total: {filtered_df['Traffic Volume'].sum():,.0f} vehicles</div>
            </div>""", unsafe_allow_html=True)
        with k4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Mean Vehicular Speed</div>
                <div class="kpi-value">{mean_speed:.1f} km/h</div>
                <div class="kpi-sub">Range: {filtered_df['Average Speed'].min():.1f}–{filtered_df['Average Speed'].max():.1f} km/h</div>
            </div>""", unsafe_allow_html=True)
        with k5:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Incident Reports</div>
                <div class="kpi-value">{total_inc:,}</div>
                <div class="kpi-sub">Avg {filtered_df['Incident Reports'].mean():.2f} / road-day</div>
            </div>""", unsafe_allow_html=True)

        # Row 1: Target Class Distribution & Sector-Level Congestion Rate
        st.markdown('<div class="section-header">1. Target Class Balance & Spatial Distribution Across Bangalore Sectors</div>', unsafe_allow_html=True)
        r1c1, r1c2 = st.columns([1, 1.55])

        with r1c1:
            class_counts = (
                filtered_df["Congestion_Class_Label"]
                .value_counts()
                .reindex(["Moderate/Low (<80%)", "High Congestion (≥80%)"], fill_value=0)
                .reset_index()
            )
            class_counts.columns = ["Congestion State", "Count"]
            class_counts["Percentage"] = (class_counts["Count"] / class_counts["Count"].sum() * 100).round(1)

            fig_target = px.bar(
                class_counts,
                x="Congestion State",
                y="Count",
                color="Congestion State",
                text=class_counts["Percentage"].apply(lambda x: f"{x:.1f}%"),
                color_discrete_map={
                    "Moderate/Low (<80%)": "#86EFAC",
                    "High Congestion (≥80%)": "#14532D",
                },
                title="Target Distribution: High (≥80%) vs. Moderate/Low (<80%)"
            )
            fig_target.update_traces(textposition="outside", marker_line_color="#14532D", marker_line_width=1)
            fig_target.update_layout(
                plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                font=dict(color="#143321"), showlegend=False,
                yaxis_title="Number of Road-Day Observations", xaxis_title="",
                margin=dict(t=45, b=20, l=20, r=20), height=370
            )
            st.plotly_chart(fig_target, use_container_width=True)

        with r1c2:
            area_summary = (
                filtered_df.groupby("Area Name", as_index=False)
                .agg(
                    High_Rate=("high_congestion", lambda s: s.mean() * 100),
                    Observations=("high_congestion", "count"),
                    Avg_Volume=("Traffic Volume", "mean")
                )
                .sort_values("High_Rate", ascending=False)
            )
            city_avg = filtered_df["high_congestion"].mean() * 100

            fig_area = px.bar(
                area_summary,
                x="Area Name",
                y="High_Rate",
                color="High_Rate",
                color_continuous_scale=["#A7F3D0", "#16A34A", "#14532D"],
                text=area_summary["High_Rate"].apply(lambda v: f"{v:.1f}%"),
                hover_data={"Observations": True, "Avg_Volume": ":,.0f"},
                title="Percentage of High Congestion Days by Urban Sector (vs. Baseline)"
            )
            fig_area.add_hline(
                y=city_avg,
                line_dash="dash",
                line_color="#B91C1C",
                annotation_text=f"Mean Baseline ({city_avg:.1f}%)",
                annotation_position="top right"
            )
            fig_area.update_traces(textposition="outside")
            fig_area.update_layout(
                plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                font=dict(color="#143321"), coloraxis_showscale=False,
                yaxis_title="% High Congestion Days (≥80%)", xaxis_title="Urban Sector",
                yaxis_range=[0, min(105, max(area_summary["High_Rate"].max() * 1.18, 75))],
                margin=dict(t=45, b=20, l=20, r=20), height=370
            )
            st.plotly_chart(fig_area, use_container_width=True)

        # Row 2: Speed & Volume Separation by Congestion Class
        st.markdown('<div class="section-header">2. Physical Traffic Drivers: Vehicular Speed & Volume Class Separation</div>', unsafe_allow_html=True)
        r2c1, r2c2 = st.columns(2)

        with r2c1:
            fig_speed_box = px.box(
                filtered_df,
                x="Congestion_Class_Label",
                y="Average Speed",
                color="Congestion_Class_Label",
                category_orders={"Congestion_Class_Label": ["Moderate/Low (<80%)", "High Congestion (≥80%)"]},
                color_discrete_map={
                    "Moderate/Low (<80%)": "#4ADE80",
                    "High Congestion (≥80%)": "#14532D",
                },
                title="Average Vehicular Speed (km/h) by Congestion Class"
            )
            fig_speed_box.update_layout(
                plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                font=dict(color="#143321"), showlegend=False,
                xaxis_title="Congestion State", yaxis_title="Average Speed (km/h)",
                margin=dict(t=45, b=20, l=20, r=20), height=370
            )
            st.plotly_chart(fig_speed_box, use_container_width=True)

        with r2c2:
            fig_vol_box = px.box(
                filtered_df,
                x="Congestion_Class_Label",
                y="Traffic Volume",
                color="Congestion_Class_Label",
                category_orders={"Congestion_Class_Label": ["Moderate/Low (<80%)", "High Congestion (≥80%)"]},
                color_discrete_map={
                    "Moderate/Low (<80%)": "#4ADE80",
                    "High Congestion (≥80%)": "#14532D",
                },
                title="Daily Traffic Volume by Congestion Class"
            )
            fig_vol_box.update_layout(
                plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                font=dict(color="#143321"), showlegend=False,
                xaxis_title="Congestion State", yaxis_title="Traffic Volume (Vehicles)",
                margin=dict(t=45, b=20, l=20, r=20), height=370
            )
            st.plotly_chart(fig_vol_box, use_container_width=True)

        # Row 3: Temporal & Weather Analysis + Monthly Trajectory
        st.markdown('<div class="section-header">3. Temporal, Weather & Corridor Bottleneck Patterns</div>', unsafe_allow_html=True)
        r3c1, r3c2, r3c3 = st.columns(3)

        with r3c1:
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            day_df = (
                filtered_df.groupby("DayOfWeek", as_index=False)["high_congestion"]
                .mean()
                .assign(High_Rate=lambda d: d["high_congestion"] * 100)
            )
            day_df["DayOfWeek"] = pd.Categorical(day_df["DayOfWeek"], categories=day_order, ordered=True)
            day_df = day_df.sort_values("DayOfWeek")

            fig_day = px.bar(
                day_df,
                x="DayOfWeek",
                y="High_Rate",
                color="High_Rate",
                color_continuous_scale=["#BBF7D0", "#16A34A", "#14532D"],
                text=day_df["High_Rate"].apply(lambda v: f"{v:.1f}%"),
                title="High Congestion Rate by Day of Week"
            )
            fig_day.update_traces(textposition="outside")
            fig_day.update_layout(
                plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                font=dict(color="#143321"), coloraxis_showscale=False,
                xaxis_title="", yaxis_title="% High Congestion",
                yaxis_range=[0, 85], margin=dict(t=45, b=20, l=20, r=20), height=350
            )
            st.plotly_chart(fig_day, use_container_width=True)

        with r3c2:
            weather_df = (
                filtered_df.groupby("Weather Conditions", as_index=False)["high_congestion"]
                .agg(High_Rate=lambda s: s.mean() * 100, Count="count")
                .sort_values("High_Rate", ascending=False)
            )
            fig_wea = px.bar(
                weather_df,
                x="Weather Conditions",
                y="High_Rate",
                color="High_Rate",
                color_continuous_scale=["#BBF7D0", "#16A34A", "#14532D"],
                text=weather_df["High_Rate"].apply(lambda v: f"{v:.1f}%"),
                hover_data=["Count"],
                title="High Congestion Rate by Weather Condition"
            )
            fig_wea.update_traces(textposition="outside")
            fig_wea.update_layout(
                plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                font=dict(color="#143321"), coloraxis_showscale=False,
                xaxis_title="", yaxis_title="% High Congestion",
                yaxis_range=[0, 85], margin=dict(t=45, b=20, l=20, r=20), height=350
            )
            st.plotly_chart(fig_wea, use_container_width=True)

        with r3c3:
            road_df = (
                filtered_df.groupby("Road/Intersection Name", as_index=False)["high_congestion"]
                .agg(High_Rate=lambda s: s.mean() * 100, Volume="count")
                .sort_values("High_Rate", ascending=True)
                .tail(10)
            )
            fig_road = px.bar(
                road_df,
                x="High_Rate",
                y="Road/Intersection Name",
                orientation="h",
                color="High_Rate",
                color_continuous_scale=["#A7F3D0", "#16A34A", "#14532D"],
                text=road_df["High_Rate"].apply(lambda v: f"{v:.1f}%"),
                title="Top Intersections by High Congestion Rate (%)"
            )
            fig_road.update_traces(textposition="outside")
            fig_road.update_layout(
                plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                font=dict(color="#143321"), coloraxis_showscale=False,
                xaxis_title="% High Congestion", yaxis_title="",
                xaxis_range=[0, 108], margin=dict(t=45, b=20, l=20, r=20), height=350
            )
            st.plotly_chart(fig_road, use_container_width=True)

        # Row 4: Monthly Trend & Correlation Matrix
        st.markdown('<div class="section-header">4. Monthly Traffic Trend & Numeric Feature Correlation Audit</div>', unsafe_allow_html=True)
        r4c1, r4c2 = st.columns([1.2, 1])

        with r4c1:
            monthly_trend = (
                filtered_df.assign(YearMonth=filtered_df["Parsed_Date"].dt.to_period("M").astype(str))
                .groupby("YearMonth", as_index=False)
                .agg(
                    Avg_Volume=("Traffic Volume", "mean"),
                    High_Cong_Pct=("high_congestion", lambda s: s.mean() * 100)
                )
            )
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=monthly_trend["YearMonth"],
                y=monthly_trend["Avg_Volume"],
                name="Mean Traffic Volume",
                mode="lines+markers",
                line=dict(color="#14532D", width=2.5),
                marker=dict(size=6, color="#14532D")
            ))
            fig_trend.update_layout(
                title="Monthly Mean Traffic Volume Trajectory (Jan 2022 – Aug 2024)",
                plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                font=dict(color="#143321"),
                xaxis_title="Month", yaxis_title="Mean Daily Vehicles per Intersection",
                margin=dict(t=45, b=20, l=20, r=20), height=400
            )
            st.plotly_chart(fig_trend, use_container_width=True)

        with r4c2:
            corr_cols = [
                "Traffic Volume", "Average Speed", "Incident Reports",
                "Pedestrian and Cyclist Count", "Congestion Level",
                "Road Capacity Utilization", "Travel Time Index", "Environmental Impact",
                "high_congestion"
            ]
            corr_mat = filtered_df[corr_cols].corr().round(2)
            fig_corr = px.imshow(
                corr_mat,
                text_auto=True,
                color_continuous_scale=["#FEE2E2", "#FFFFFF", "#16A34A", "#14532D"],
                zmin=-1, zmax=1,
                title="Pearson Correlation Matrix (Exposing Leakage vs. Genuine Predictors)"
            )
            fig_corr.update_layout(
                plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                font=dict(color="#143321", size=10),
                margin=dict(t=45, b=20, l=20, r=20), height=400
            )
            st.plotly_chart(fig_corr, use_container_width=True)


# ===========================================================================
# TAB 2: PREDICTIVE CONGESTION SIMULATOR (LIVE ML)
# ===========================================================================
with tab_pred:
    st.markdown("""
    <div class="info-box-green">
        <strong>🤖 Interactive Congestion Severity Classifier (Deliverable 2 & 3 Integration):</strong>
        Configure road, traffic, weather, and temporal parameters below—or click one of the <strong>Quick Preset Scenarios</strong>—to run real-time inference across all <strong>5 trained Machine Learning Pipelines</strong> (fitted strictly on the 7,148-row training set using the 11 leak-free features).
    </div>
    """, unsafe_allow_html=True)

    # Preset Scenario State Initialization
    if "sim_area" not in st.session_state:
        st.session_state.sim_area = "Koramangala"
        st.session_state.sim_road = "Sony World Junction"
        st.session_state.sim_volume = 42000
        st.session_state.sim_speed = 31.0
        st.session_state.sim_incidents = 2
        st.session_state.sim_peds = 105
        st.session_state.sim_weather = "Clear"
        st.session_state.sim_roadwork = "No"
        st.session_state.sim_month = 6
        st.session_state.sim_day = "Wednesday"

    # Quick Preset Buttons
    st.markdown("##### ⚡ Quick Demo Scenarios (One-Click Load)")
    pcol1, pcol2, pcol3, pcol4 = st.columns(4)
    if pcol1.button("🚨 Peak Corridor Gridlock", use_container_width=True):
        st.session_state.sim_area = "Koramangala"
        st.session_state.sim_road = "Sony World Junction"
        st.session_state.sim_volume = 52000
        st.session_state.sim_speed = 26.5
        st.session_state.sim_incidents = 3
        st.session_state.sim_peds = 115
        st.session_state.sim_weather = "Rain"
        st.session_state.sim_roadwork = "Yes"
        st.session_state.sim_month = 8
        st.session_state.sim_day = "Friday"
    if pcol2.button("🌿 Free-Flowing Traffic", use_container_width=True):
        st.session_state.sim_area = "Whitefield"
        st.session_state.sim_road = "Marathahalli Bridge"
        st.session_state.sim_volume = 11500
        st.session_state.sim_speed = 52.0
        st.session_state.sim_incidents = 0
        st.session_state.sim_peds = 160
        st.session_state.sim_weather = "Clear"
        st.session_state.sim_roadwork = "No"
        st.session_state.sim_month = 2
        st.session_state.sim_day = "Sunday"
    if pcol3.button("⚠️ Borderline Transition Zone", use_container_width=True):
        st.session_state.sim_area = "Jayanagar"
        st.session_state.sim_road = "South End Circle"
        st.session_state.sim_volume = 23500
        st.session_state.sim_speed = 38.0
        st.session_state.sim_incidents = 1
        st.session_state.sim_peds = 98
        st.session_state.sim_weather = "Overcast"
        st.session_state.sim_roadwork = "No"
        st.session_state.sim_month = 10
        st.session_state.sim_day = "Tuesday"
    if pcol4.button("🔄 Reset to Dataset Median", use_container_width=True):
        st.session_state.sim_area = "Indiranagar"
        st.session_state.sim_road = "100 Feet Road"
        st.session_state.sim_volume = int(df["Traffic Volume"].median())
        st.session_state.sim_speed = float(round(df["Average Speed"].median(), 1))
        st.session_state.sim_incidents = int(df["Incident Reports"].median())
        st.session_state.sim_peds = int(df["Pedestrian and Cyclist Count"].median())
        st.session_state.sim_weather = "Clear"
        st.session_state.sim_roadwork = "No"
        st.session_state.sim_month = 5
        st.session_state.sim_day = "Monday"

    st.markdown("---")
    left_ctrl, right_out = st.columns([1.15, 1.35])

    with left_ctrl:
        st.markdown('<div class="section-header">🎛️ Input Road-Day Observation (11 Features)</div>', unsafe_allow_html=True)

        ml_models_list = [
            "Random Forest",
            "Logistic Regression",
            "Support Vector Machine (RBF)",
            "K-Nearest Neighbors (KNN)",
            "Gaussian Naive Bayes",
        ]
        m_col1, m_col2 = st.columns([1.4, 1])
        with m_col1:
            active_model_name = st.selectbox(
                "🎯 Primary Classification Algorithm",
                ml_models_list,
                index=0,
                help="Random Forest (97.30% Recall) and Logistic Regression (95.08% Accuracy) are co-recommended."
            )
        with m_col2:
            decision_threshold = st.slider(
                "⚖️ Alert Threshold",
                min_value=0.20, max_value=0.80, value=0.50, step=0.05,
                help="Lower threshold increases High-Congestion Recall (fewer missed gridlock days)."
            )

        c_sp1, c_sp2 = st.columns(2)
        area_list = sorted(df["Area Name"].unique().tolist())
        with c_sp1:
            in_area = st.selectbox(
                "1. Urban Sector (Area Name)",
                area_list,
                index=area_list.index(st.session_state.sim_area) if st.session_state.sim_area in area_list else 0
            )
        area_roads = sorted(df.loc[df["Area Name"] == in_area, "Road/Intersection Name"].unique().tolist())
        default_road_idx = area_roads.index(st.session_state.sim_road) if st.session_state.sim_road in area_roads else 0
        with c_sp2:
            in_road = st.selectbox("2. Road / Intersection Name", area_roads, index=default_road_idx)

        c_num1, c_num2 = st.columns(2)
        with c_num1:
            in_volume = st.slider(
                "3. Traffic Volume (Vehicles/Day)",
                min_value=4000, max_value=75000, value=int(st.session_state.sim_volume), step=500
            )
            in_incidents = st.slider(
                "5. Incident Reports (Count)",
                min_value=0, max_value=10, value=int(st.session_state.sim_incidents), step=1
            )
        with c_num2:
            in_speed = st.slider(
                "4. Average Speed (km/h)",
                min_value=20.0, max_value=90.0, value=float(st.session_state.sim_speed), step=0.5
            )
            in_peds = st.slider(
                "6. Pedestrian & Cyclist Count",
                min_value=50, max_value=260, value=int(st.session_state.sim_peds), step=2
            )

        c_ctx1, c_ctx2 = st.columns(2)
        weather_list = sorted(df["Weather Conditions"].unique().tolist())
        roadwork_list = ["No", "Yes"]
        with c_ctx1:
            in_weather = st.selectbox(
                "7. Weather Conditions",
                weather_list,
                index=weather_list.index(st.session_state.sim_weather) if st.session_state.sim_weather in weather_list else 0
            )
            in_month = st.slider(
                "9. Calendar Month (1–12)",
                min_value=1, max_value=12, value=int(st.session_state.sim_month), step=1
            )
        day_list = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        with c_ctx2:
            in_roadwork = st.selectbox(
                "8. Roadwork & Construction Activity",
                roadwork_list,
                index=roadwork_list.index(st.session_state.sim_roadwork) if st.session_state.sim_roadwork in roadwork_list else 0
            )
            in_day = st.selectbox(
                "10. Day of Week",
                day_list,
                index=day_list.index(st.session_state.sim_day) if st.session_state.sim_day in day_list else 0
            )

        in_is_weekend = 1 if in_day in ["Saturday", "Sunday"] else 0
        st.caption(f"**11. Engineered Weekend Indicator (`IsWeekend`):** `{in_is_weekend}` ({'Weekend' if in_is_weekend else 'Weekday'})")

    # Build Single-Row Input DataFrame Matching Exact Training Schema
    input_df = pd.DataFrame([{
        "Traffic Volume": in_volume,
        "Average Speed": in_speed,
        "Incident Reports": in_incidents,
        "Pedestrian and Cyclist Count": in_peds,
        "Month": in_month,
        "IsWeekend": in_is_weekend,
        "Area Name": in_area,
        "Road/Intersection Name": in_road,
        "Weather Conditions": in_weather,
        "Roadwork and Construction Activity": in_roadwork,
        "DayOfWeek": in_day,
    }])

    with right_out:
        st.markdown('<div class="section-header">📡 Live Prediction & Multi-Model Consensus</div>', unsafe_allow_html=True)

        primary_pipe = artifacts["pipelines"][active_model_name]
        primary_prob = float(primary_pipe.predict_proba(input_df)[0, 1])
        primary_pred = int(primary_prob >= decision_threshold)

        if primary_pred == 1:
            st.markdown(f"""
            <div class="pred-card-high">
                <div style="font-size:0.85rem; font-weight:700; text-transform:uppercase; letter-spacing:0.6px;">
                    🚨 Predicted Classification ({active_model_name})
                </div>
                <div style="font-size:1.75rem; font-weight:900; margin:0.25rem 0;">
                    HIGH CONGESTION (≥ 80% Severity)
                </div>
                <div style="font-size:0.95rem;">
                    Predicted Probability: <strong>{primary_prob*100:.1f}%</strong> (Alert Threshold: {decision_threshold*100:.0f}%) ·
                    Corridor: <strong>{in_road} ({in_area})</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="pred-card-low">
                <div style="font-size:0.85rem; font-weight:700; text-transform:uppercase; letter-spacing:0.6px;">
                    🌿 Predicted Classification ({active_model_name})
                </div>
                <div style="font-size:1.75rem; font-weight:900; margin:0.25rem 0;">
                    MODERATE / LOW CONGESTION (&lt; 80% Severity)
                </div>
                <div style="font-size:0.95rem;">
                    Predicted Probability of High Congestion: <strong>{primary_prob*100:.1f}%</strong> (Alert Threshold: {decision_threshold*100:.0f}%) ·
                    Corridor: <strong>{in_road} ({in_area})</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Probability Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=primary_prob * 100,
            number={"suffix": "%", "font": {"size": 34, "color": "#14532D"}},
            title={"text": f"High-Congestion Risk Probability ({active_model_name})", "font": {"size": 14, "color": "#143321"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#14532D"},
                "bar": {"color": "#B91C1C" if primary_pred == 1 else "#16A34A"},
                "bgcolor": "#FFFFFF",
                "borderwidth": 1,
                "bordercolor": "#CBEAD7",
                "steps": [
                    {"range": [0, decision_threshold * 100], "color": "#DCFCE7"},
                    {"range": [decision_threshold * 100, 100], "color": "#FEE2E2"},
                ],
                "threshold": {
                    "line": {"color": "#14532D", "width": 4},
                    "thickness": 0.8,
                    "value": decision_threshold * 100,
                },
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="#FFFFFF", font=dict(color="#143321"),
            margin=dict(t=40, b=10, l=25, r=25), height=225
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Consensus Across All 5 Models
        consensus_rows = []
        for m_name in ml_models_list:
            m_pipe = artifacts["pipelines"][m_name]
            m_prob = float(m_pipe.predict_proba(input_df)[0, 1])
            m_pred = "🚨 High Congestion (1)" if m_prob >= decision_threshold else "🌿 Moderate/Low (0)"
            consensus_rows.append({
                "Algorithm": m_name,
                "High Congestion Probability (%)": round(m_prob * 100, 2),
                "Predicted Class": m_pred,
            })
        consensus_df = pd.DataFrame(consensus_rows)
        high_votes = sum(1 for r in consensus_rows if "High" in r["Predicted Class"])

        st.markdown(f"**Model Ensemble Consensus:** `{high_votes} / 5` algorithms flag this scenario as **High Congestion**.")
        st.dataframe(consensus_df, use_container_width=True, hide_index=True)

        # Recommended Action for Target Audience
        if primary_pred == 1:
            st.markdown("""
            <div class="info-box-green">
                <strong>🚦 Recommended Municipal Traffic Action (High Risk Alert):</strong><br>
                • <strong>Signal Operations:</strong> Activate green-wave progression and extended green phases along the selected corridor.<br>
                • <strong>Commuter Advisory:</strong> Broadcast proactive congestion alerts and suggest alternate arterial routes.<br>
                • <strong>Incident Response:</strong> Pre-position rapid-clearance tow units if incident counts ≥ 2 or roadwork is active.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box-green">
                <strong>✅ Recommended Municipal Traffic Action (Manageable Flow):</strong><br>
                • Maintain standard fixed-time or adaptive signal cycles.<br>
                • No emergency rerouting required; monitor corridor sensors for sudden volume spikes above 30,000 vehicles/day.
            </div>
            """, unsafe_allow_html=True)


# ===========================================================================
# TAB 3: MODEL BENCHMARKING & DIAGNOSTICS (DELIVERABLE 3)
# ===========================================================================
with tab_eval:
    st.markdown('<div class="section-header">1. Benchmark Evaluation on Held-Out Test Set (n = 1,788 Stratified Split)</div>', unsafe_allow_html=True)
    st.markdown("""
    All five supervised classification algorithms and the **Majority Baseline (Always High = 62.2%)** were trained strictly on the **7,148-row training set** with 5-fold stratified cross-validation and evaluated on the unseen **1,788-row test set** (`1,112` High Congestion rows, `676` Moderate/Low rows). **High-Congestion Recall** is the headline operational metric.
    """)

    st.dataframe(artifacts["metrics_df"], use_container_width=True, hide_index=True)

    e_col1, e_col2 = st.columns(2)
    with e_col1:
        ml_only_df = artifacts["metrics_df"][artifacts["metrics_df"]["Model"] != "Majority Baseline (Always High)"].copy()
        melted_metrics = ml_only_df.melt(
            id_vars="Model",
            value_vars=["Test Accuracy (%)", "Test Precision (%)", "Test Recall (%)"],
            var_name="Metric",
            value_name="Score (%)"
        )
        fig_bench = px.bar(
            melted_metrics,
            x="Model",
            y="Score (%)",
            color="Metric",
            barmode="group",
            color_discrete_map={
                "Test Accuracy (%)": "#14532D",
                "Test Precision (%)": "#16A34A",
                "Test Recall (%)": "#4ADE80",
            },
            title="Test Set Accuracy, Precision & Recall vs. 62.2% Baseline Floor"
        )
        fig_bench.add_hline(
            y=62.19, line_dash="dash", line_color="#B91C1C",
            annotation_text="Majority Baseline (62.2%)", annotation_position="bottom right"
        )
        fig_bench.update_layout(
            plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
            font=dict(color="#143321"), yaxis_range=[50, 103],
            xaxis_title="", margin=dict(t=45, b=20, l=20, r=20), height=410
        )
        st.plotly_chart(fig_bench, use_container_width=True)

    with e_col2:
        fig_roc = go.Figure()
        roc_colors = {
            "Random Forest": "#14532D",
            "Logistic Regression": "#16A34A",
            "Support Vector Machine (RBF)": "#0D9488",
            "K-Nearest Neighbors (KNN)": "#65A30D",
            "Gaussian Naive Bayes": "#86EFAC",
        }
        for m_name, color in roc_colors.items():
            fpr, tpr, auc_val = artifacts["roc_data"][m_name]
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"{m_name} (AUC = {auc_val:.4f})",
                line=dict(color=color, width=2.5)
            ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            name="Majority Baseline (AUC = 0.5000)",
            line=dict(color="#B91C1C", dash="dash", width=1.5)
        ))
        fig_roc.update_layout(
            title="ROC Curves Across All Five Classification Models (Test Set)",
            plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
            font=dict(color="#143321"),
            xaxis_title="False Positive Rate (1 - Specificity)",
            yaxis_title="True Positive Rate (Sensitivity / Recall)",
            legend=dict(x=0.36, y=0.08, bgcolor="rgba(255,255,255,0.85)"),
            margin=dict(t=45, b=20, l=20, r=20), height=410
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # Row 2: Confusion Matrix Explorer & Feature Importance
    st.markdown('<div class="section-header">2. Confusion Matrix Explorer & Random Forest Feature Importances</div>', unsafe_allow_html=True)
    cm_col, fi_col = st.columns([1, 1.25])

    with cm_col:
        selected_cm_model = st.selectbox(
            "Select Algorithm to Inspect Confusion Matrix:",
            [m for m in artifacts["metrics_df"]["Model"] if m != "Majority Baseline (Always High)"],
            index=0
        )
        cm = artifacts["cm_data"][selected_cm_model]
        tn, fp, fn, tp = cm.ravel()
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            x=["Predicted Mod/Low (0)", "Predicted High (1)"],
            y=["Actual Mod/Low (0)", "Actual High (1)"],
            color_continuous_scale=["#F0FDF4", "#86EFAC", "#16A34A", "#14532D"],
            title=f"Confusion Matrix: {selected_cm_model} (FN = {fn} Missed High Days)"
        )
        fig_cm.update_layout(
            plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
            font=dict(color="#143321", size=12), coloraxis_showscale=False,
            margin=dict(t=45, b=20, l=20, r=20), height=350
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        st.caption(
            f"**True Positives (Caught High Days):** `{tp}` · **False Negatives (Missed High Days):** `{fn}` · "
            f"**True Negatives:** `{tn}` · **False Positives:** `{fp}`"
        )

    with fi_col:
        top_fi = artifacts["feat_imp_df"].head(12).sort_values("Importance", ascending=True)
        fig_fi = px.bar(
            top_fi,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale=["#A7F3D0", "#16A34A", "#14532D"],
            text=top_fi["Importance"].apply(lambda v: f"{v*100:.1f}%"),
            title="Top 12 Predictive Features (Random Forest Gini Importance)"
        )
        fig_fi.update_traces(textposition="outside")
        fig_fi.update_layout(
            plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
            font=dict(color="#143321"), coloraxis_showscale=False,
            xaxis_title="Relative Importance Share", yaxis_title="",
            xaxis_range=[0, top_fi["Importance"].max() * 1.18],
            margin=dict(t=45, b=20, l=20, r=20), height=400
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    # Row 3: Robustness & Sensitivity Checks from Project Foundation PDF
    st.markdown('<div class="section-header">3. Robustness & Sensitivity Checks (Ablation & Chronological Split)</div>', unsafe_allow_html=True)
    diag1, diag2 = st.columns(2)
    with diag1:
        st.markdown("##### 🧪 Check 1: Ablation Without `Traffic Volume` & `Average Speed`")
        st.caption(
            "Tests whether models still beat the 62.2% baseline when physical volume and speed sensors are excluded "
            "(using only road, area, incidents, pedestrian counts, weather, roadwork, and calendar features)."
        )
        st.dataframe(artifacts["ablation_df"], use_container_width=True, hide_index=True)
    with diag2:
        st.markdown("##### ⏳ Check 2: Chronological Split (Train 2022–2023 → Test 2024)")
        st.caption(
            "Tests temporal generalization by training strictly on 2022–2023 (6,837 rows) and evaluating on future "
            "2024 observations (2,099 rows) to rule out consecutive-day autocorrelation leakage."
        )
        st.dataframe(artifacts["chrono_df"], use_container_width=True, hide_index=True)


# ===========================================================================
# TAB 4: PROJECT FOUNDATION & DATASET EXPLORER
# ===========================================================================
with tab_found:
    st.markdown('<div class="section-header">1. Project Foundation Summary (Deliverable 1 Alignment)</div>', unsafe_allow_html=True)
    fcol1, fcol2 = st.columns(2)

    with fcol1:
        st.markdown("""
        <div class="info-box-green">
            <strong>📌 Problem Statement & Scope</strong><br>
            Traffic authorities in Bangalore react to severe gridlock only after it forms. This study evaluates whether historical daily observations of vehicular volume, average speed, incidents, pedestrian/cyclist activity, road segment, area, weather, roadwork, and calendar position can classify a road-day as <strong>High Congestion (≥80%)</strong> vs. <strong>Moderate/Low (&lt;80%)</strong> against the <strong>62.2% Always-High baseline</strong>.
        </div>
        <div class="info-box-green">
            <strong>👥 Target Audience & Supported Decisions</strong><br>
            • <strong>Intended Users:</strong> Municipal traffic authorities, urban planners, and traffic engineers.<br>
            • <strong>Decisions Supported:</strong> Identifying high-risk road-day conditions for proactive signal re-timing or commuter advisories, and evaluating pilot alerting feasibility.<br>
            • <strong>Out of Scope:</strong> Causal treatment estimation or multi-day forecasting without forecasted weather/volume inputs.
        </div>
        """, unsafe_allow_html=True)

    with fcol2:
        st.markdown("""
        <div class="info-box-green">
            <strong>✅ 11 Predictors Retained for Modeling</strong><br>
            • <strong>Numerical (6):</strong> <code>Traffic Volume</code>, <code>Average Speed</code>, <code>Incident Reports</code>, <code>Pedestrian and Cyclist Count</code>, <code>Month</code>, <code>IsWeekend</code><br>
            • <strong>Categorical (5):</strong> <code>Area Name</code>, <code>Road/Intersection Name</code>, <code>Weather Conditions</code>, <code>Roadwork and Construction Activity</code>, <code>DayOfWeek</code>
        </div>
        <div class="info-box-green">
            <strong>🚫 9 Columns Dropped (Leakage, Redundancy & Noise Prevention)</strong><br>
            • <strong>Target Source:</strong> <code>Congestion Level</code> (used to define binary target ≥80%)<br>
            • <strong>Deterministic Duplicate:</strong> <code>Environmental Impact</code> ($r = 1.00$ with Traffic Volume)<br>
            • <strong>Post-Hoc Leakage:</strong> <code>Road Capacity Utilization</code> ($r = 0.865$), <code>Travel Time Index</code> ($r = 0.752$)<br>
            • <strong>Uninformative Noise ($|r| &lt; 0.03$):</strong> <code>Public Transport Usage</code>, <code>Traffic Signal Compliance</code>, <code>Parking Usage</code>, plus raw <code>Date</code> / <code>Parsed_Date</code>.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">2. Dataset Explorer & Export (Deliverable 4)</div>', unsafe_allow_html=True)
    d_choice = st.radio(
        "Select Dataset View:",
        ["Cleaned Modeling Dataset (bangalore_traffic_cleaned.csv — 12 Columns)", "Raw Dataset (Banglore_traffic_Dataset.csv — 16 Columns)"],
        horizontal=True
    )
    if "Cleaned" in d_choice:
        st.dataframe(df_clean.head(200), use_container_width=True)
        st.download_button(
            "⬇️ Download Cleaned Dataset (bangalore_traffic_cleaned.csv)",
            data=df_clean.to_csv(index=False).encode("utf-8"),
            file_name="bangalore_traffic_cleaned.csv",
            mime="text/csv"
        )
    else:
        st.dataframe(df.drop(columns=["Parsed_Date", "Year", "Month", "DayOfWeek", "IsWeekend", "high_congestion", "Congestion_Class_Label"]).head(200), use_container_width=True)
        st.download_button(
            "⬇️ Download Raw Dataset (Banglore_traffic_Dataset.csv)",
            data=pd.read_csv("Banglore_traffic_Dataset.csv").to_csv(index=False).encode("utf-8"),
            file_name="Banglore_traffic_Dataset.csv",
            mime="text/csv"
        )
