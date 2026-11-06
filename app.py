
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Nexus | Customer Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 2. DESIGN SYSTEM
# =========================================================
st.markdown("""
<style>
:root { --bg:#080D18; --surface:#101827; --surface-2:#151F31; --text:#EDF3FF; --muted:#9BAAC2; --border:#28364D; --blue:#6EA8FE; }
.stApp { background:radial-gradient(ellipse at 8% 0%,#172844 0%,#0B1220 36%,var(--bg) 75%); color:var(--text); }
[data-testid="stHeader"] { background:rgba(8,13,24,.72); }
[data-testid="stAppViewContainer"] .main .block-container { max-width:1500px; padding-top:1.8rem; padding-bottom:3rem; }
[data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child { background:#0B1322; border-right:1px solid #223149; }
[data-testid="stSidebar"] * { color:#E4ECFA; }
[data-testid="stSidebar"] label { color:#BDCAE0 !important; }
[data-testid="stSidebar"] hr { border-color:#26354D; }
[data-testid="stSidebar"] [data-testid="stFileUploader"] { background:#111D30; border:1px solid #2C405F; border-radius:14px; padding:10px; }
[data-testid="stSidebar"] [data-testid="stFileUploader"] section { background:#111D30; border:none; }
[data-testid="stSidebar"] [data-testid="stFileUploader"] button { background:#477FE8; color:white; border:0; }
[data-testid="stSidebar"] [data-testid="stRadio"] > div { gap:6px; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { padding:9px 11px; border-radius:10px; border:1px solid transparent; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background:#16243A; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) { background:#1A2D4B; border-color:#365C91; }
.brand { display:flex; align-items:center; gap:12px; padding:10px 2px 28px; }
.brand-mark { width:43px; height:43px; display:flex; align-items:center; justify-content:center; border-radius:14px; background:linear-gradient(135deg,#73B5FF,#7C6CF2); color:#081120; font-size:23px; font-weight:900; box-shadow:0 8px 26px #477FE833; }
.brand-name { color:#F4F7FF; font-size:23px; line-height:1.1; font-weight:800; letter-spacing:-.8px; }
.brand-caption { color:#91A6C7; font-size:10px; letter-spacing:1.25px; text-transform:uppercase; margin-top:5px; }
.workspace-header { display:flex; justify-content:space-between; align-items:flex-start; gap:20px; padding:27px 30px; margin:0 0 23px; border:1px solid #2A3A55; border-radius:22px; background:linear-gradient(120deg,rgba(26,43,68,.96),rgba(15,24,40,.94)); box-shadow:0 18px 50px #00000020; }
.eyebrow { color:#77B9FF; font-size:10px; letter-spacing:2px; font-weight:800; margin-bottom:10px; }
.page-title { color:#F4F7FF; font-size:31px; line-height:1.16; font-weight:800; letter-spacing:-1px; }
.page-description { color:#AAB8CE; font-size:13px; line-height:1.65; margin-top:9px; max-width:780px; }
.status-pill { white-space:nowrap; padding:8px 12px; border-radius:99px; color:#8FF0C0; background:#143A32; border:1px solid #28604F; font-size:11px; font-weight:700; }
.metric-card { min-height:142px; padding:19px 20px; border-radius:17px; border:1px solid #2B3B54; background:linear-gradient(145deg,#151F31,#101827); box-shadow:0 10px 28px #00000018; position:relative; overflow:hidden; }
.metric-card:before { content:""; position:absolute; left:0; top:0; bottom:0; width:3px; background:linear-gradient(180deg,#72B4FF,#8C82FF); }
.metric-top { display:flex; justify-content:space-between; align-items:center; gap:10px; }
.metric-label { color:#9CAEC8; font-size:10px; letter-spacing:1.2px; font-weight:800; }
.metric-icon { color:#8BC3FF; font-size:19px; }
.metric-value { color:#F2F6FF; font-size:31px; line-height:1.2; font-weight:800; letter-spacing:-.8px; margin-top:15px; }
.metric-note { color:#9BAAC2; font-size:11px; margin-top:7px; line-height:1.45; }
.panel { padding:23px; margin:0 0 18px; background:rgba(16,24,39,.88); border:1px solid #28374F; border-radius:18px; box-shadow:0 12px 34px #00000018; }
.panel-heading { color:#EDF3FF; font-size:16px; font-weight:750; letter-spacing:-.2px; margin-bottom:5px; }
.panel-subtitle { color:#9BAAC2; font-size:12px; line-height:1.55; margin-bottom:17px; }
.segment-card { height:100%; min-height:188px; padding:19px; border-radius:16px; border:1px solid #2B3B54; background:linear-gradient(150deg,#17243A,#111A2A); margin-bottom:14px; }
.segment-label { color:#79B8FF; font-size:10px; font-weight:800; letter-spacing:1.4px; }
.segment-title { color:#F2F6FF; font-size:15px; line-height:1.4; font-weight:750; margin-top:10px; }
.segment-count { color:#F4F7FF; font-size:27px; font-weight:800; margin-top:11px; }
.segment-caption { color:#9BAAC2; font-size:11px; margin-top:0; }
.segment-details { border-top:1px solid #2B3B54; color:#B7C5DA; font-size:11px; line-height:1.8; margin-top:13px; padding-top:10px; }
.insight-row { display:flex; gap:13px; align-items:flex-start; padding:13px 0; border-bottom:1px solid #26344A; color:#D5DFEF; font-size:12px; line-height:1.6; }
.insight-row:last-child { border-bottom:0; }
.insight-number { min-width:27px; height:27px; display:flex; align-items:center; justify-content:center; border-radius:9px; background:#1A3150; color:#8FC5FF; font-size:10px; font-weight:800; }
.recommendation { border:1px solid #2B3D58; border-radius:13px; background:#131F32; padding:15px 16px; margin:8px 0; }
.recommendation-title { color:#EAF2FF; font-weight:750; font-size:13px; margin-bottom:5px; }
.recommendation-copy { color:#AEBED4; font-size:12px; line-height:1.6; }
.tag { display:inline-block; padding:4px 8px; border-radius:99px; color:#9BD8FF; background:#18324D; font-size:10px; font-weight:700; margin:0 5px 6px 0; }
.empty-state { border:1px dashed #354863; border-radius:18px; padding:35px; background:#101827; }
.stMarkdown, p, li { color:inherit; }
[data-testid="stWidgetLabel"] p, label, .stCaption { color:#B7C5DA !important; }
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input, [data-testid="stSelectbox"] div[data-baseweb="select"], [data-testid="stMultiSelect"] div[data-baseweb="select"] { background:#111D30; color:#EDF3FF; border-color:#334762; }
[data-baseweb="popover"] { background:#111D30; }
[data-testid="stDataFrame"], [data-testid="stTable"] { border:1px solid #2B3B54; border-radius:12px; overflow:hidden; }
[data-testid="stAlert"] { border-radius:12px; }
.stButton > button, .stDownloadButton > button { border-radius:10px; border:1px solid #3C5C86; background:#1D3A61; color:#EDF5FF; font-weight:650; }
.stButton > button:hover, .stDownloadButton > button:hover { border-color:#76B6FF; background:#264A78; color:white; }
[data-testid="stExpander"] { border:1px solid #2B3B54; border-radius:12px; background:#111A2A; }
hr { border-color:#2B3A51; } footer { visibility:hidden; }
@media(max-width:800px) { .page-title{font-size:25px;} .workspace-header{padding:20px;flex-direction:column;} .metric-card{padding:15px;min-height:125px;} .metric-value{font-size:25px;} .block-container{padding-left:1rem;padding-right:1rem;} .panel{padding:16px;} }
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. REUSABLE UI FUNCTIONS
# =========================================================
def show_metric(label, value, note, icon):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-top">
            <div class="metric-label">{label}</div>
            <div class="metric-icon">{icon}</div>
        </div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)


def show_section(title, subtitle=""):
    st.markdown(
        f'<div class="panel-heading">{title}</div>',
        unsafe_allow_html=True
    )
    if subtitle:
        st.markdown(
            f'<div class="panel-subtitle">{subtitle}</div>',
            unsafe_allow_html=True
        )


def show_header(title, description, status=None):
    pill = (
        f'<div class="status-pill">● {status}</div>'
        if status else ""
    )
    st.markdown(f"""
    <div class="workspace-header">
        <div>
            <div class="eyebrow">NEXUS / CUSTOMER INTELLIGENCE</div>
            <div class="page-title">{title}</div>
            <div class="page-description">{description}</div>
        </div>
        {pill}
    </div>
    """, unsafe_allow_html=True)


def to_csv(frame):
    return frame.to_csv(index=False).encode("utf-8-sig")


def normalized(value):
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())


def find_column(columns, aliases):
    lookup = {normalized(c): c for c in columns}
    for alias in aliases:
        if normalized(alias) in lookup:
            return lookup[normalized(alias)]
    return None


def build_segment_names(cluster_means, features, income_col, spending_col,
                        income_median, spending_median, age_col=None):
    """Create readable segment names and guarantee that every name is unique."""
    names = {}
    for cid in cluster_means.index:
        profile = cluster_means.loc[cid]
        parts = []
        if income_col:
            parts.append("Higher Income" if profile[income_col] >= income_median else "Lower Income")
        if spending_col:
            parts.append("High Spending" if profile[spending_col] >= spending_median else "Low Spending")
        names[int(cid)] = " / ".join(parts) if parts else f"Customer Group {int(cid)}"

    # If several clusters share the same income/spending archetype, distinguish
    # them using a real profile feature (age first, then another selected feature).
    groups = {}
    for cid, name in names.items():
        groups.setdefault(name, []).append(cid)

    for base_name, ids in groups.items():
        if len(ids) < 2:
            continue
        candidate_features = []
        if age_col and age_col in cluster_means.columns:
            candidate_features.append(age_col)
        candidate_features.extend(f for f in features if f not in candidate_features and f not in {income_col, spending_col})

        qualifier_feature = None
        if candidate_features:
            # Prefer the feature with the largest spread between these clusters.
            qualifier_feature = max(
                candidate_features,
                key=lambda f: float(cluster_means.loc[ids, f].max() - cluster_means.loc[ids, f].min()),
            )

        if qualifier_feature and cluster_means.loc[ids, qualifier_feature].nunique() > 1:
            midpoint = float(cluster_means[qualifier_feature].median())
            for cid in ids:
                value = float(cluster_means.loc[cid, qualifier_feature])
                if qualifier_feature == age_col:
                    qualifier = "Younger Profile" if value < midpoint else "Older Profile"
                else:
                    qualifier = f"Lower {qualifier_feature}" if value < midpoint else f"Higher {qualifier_feature}"
                names[cid] = f"{base_name} · {qualifier}"
        # Absolute fallback for genuinely indistinguishable profiles.
        seen = set()
        for cid in ids:
            if names[cid] in seen:
                names[cid] = f"{base_name} · Cohort {cid:02d}"
            seen.add(names[cid])

    return names


def scatter_chart(frame, x_feature, y_feature, height=380):
    """Plot raw feature values explicitly; scaling is used for modeling only."""
    chart_frame = frame.copy()
    chart_frame["Cluster"] = chart_frame["Cluster"].astype(str)
    tooltip_fields = [alt.Tooltip("Cluster:N", title="Cluster")]
    for field in [x_feature, y_feature, "Segment"]:
        if field in chart_frame.columns and field not in {x_feature, y_feature}:
            tooltip_fields.append(alt.Tooltip(f"{field}:N", title=field))
    tooltip_fields.extend([
        alt.Tooltip(f"{x_feature}:Q", title=x_feature, format=".2f"),
        alt.Tooltip(f"{y_feature}:Q", title=y_feature, format=".2f"),
    ])
    return alt.Chart(chart_frame).mark_circle(size=90, opacity=0.82).encode(
        x=alt.X(f"{x_feature}:Q", title=x_feature, scale=alt.Scale(zero=False)),
        y=alt.Y(f"{y_feature}:Q", title=y_feature, scale=alt.Scale(zero=False)),
        color=alt.Color("Cluster:N", title="Cluster"),
        tooltip=tooltip_fields,
    ).properties(height=height).interactive()


def build_comparison(scaled_data, max_k):
    results = []

    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(scaled_data)

        score = np.nan
        if 1 < len(np.unique(labels)) < len(labels):
            score = silhouette_score(scaled_data, labels)

        results.append({
            "K": k,
            "Inertia": float(model.inertia_),
            "Silhouette Score": float(score),
        })

    return pd.DataFrame(results)


# =========================================================
# 4. NAVIGATION RAIL
# =========================================================
with st.sidebar:
    st.markdown("""
    <div class="brand">
        <div class="brand-mark">N</div>
        <div>
            <div class="brand-name">nexus</div>
            <div class="brand-caption">Customer intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div style="color:#8FA5C3;font-size:10px;'
        'font-weight:800;letter-spacing:1.5px;margin-bottom:9px;">'
        'WORKSPACE</div>',
        unsafe_allow_html=True
    )

    page = st.radio(
        "Workspace navigation",
        [
            "Overview",
            "Cluster Analysis",
            "Customer Explorer",
            "Reports & Data",
            "Marketing Insights",
        ],
        format_func=lambda x: {
            "Overview": "◫   Overview",
            "Cluster Analysis": "◉   Cluster analysis",
            "Customer Explorer": "⌕   Customer explorer",
            "Reports & Data": "▤   Reports & data",
            "Marketing Insights": "✦   Marketing insights",
        }[x],
        label_visibility="collapsed",
        key="workspace_navigation",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown(
        '<div style="color:#8FA5C3;font-size:10px;'
        'font-weight:800;letter-spacing:1.5px;margin-bottom:10px;">'
        'DATA SOURCE</div>',
        unsafe_allow_html=True
    )

    uploaded = st.file_uploader(
        "Upload customer CSV",
        type=["csv"],
        help="Example: Mall_Customers.csv",
    )

    st.markdown("---")

    st.markdown(
        '<div style="color:#8FA5C3;font-size:10px;'
        'font-weight:800;letter-spacing:1.5px;margin-bottom:10px;">'
        'MODEL SETTINGS</div>',
        unsafe_allow_html=True
    )


# =========================================================
# 5. EMPTY STATE
# =========================================================
if uploaded is None:
    show_header(
        "Customer Intelligence",
        "A unified workspace for customer segmentation, behavior analysis, "
        "and data-driven marketing.",
    )

    st.markdown('<div class="empty-state">', unsafe_allow_html=True)
    st.markdown("## Start with your customer data")
    st.write(
        "Upload a CSV file using the left navigation panel. "
        "Nexus will validate your dataset, evaluate cluster counts, "
        "and organize customers into meaningful groups."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    a, b, c = st.columns(3)

    with a:
        show_metric("01 / DATA", "Validate", "Check records and selected features", "✓")
    with b:
        show_metric("02 / MODEL", "Discover", "Compare clusters with K-Means", "◉")
    with c:
        show_metric("03 / INSIGHTS", "Understand", "Explore customer patterns", "↗")

    st.stop()


# =========================================================
# 6. LOAD SOURCE DATA
# =========================================================
try:
    uploaded.seek(0)
    original = pd.read_csv(uploaded)
except Exception as exc:
    st.error(f"Could not read this CSV: {exc}")
    st.stop()

original.columns = original.columns.astype(str).str.strip()
original = original.loc[:, ~original.columns.duplicated()].copy()

if original.empty or len(original.columns) < 2:
    st.error("The CSV must contain data rows and at least two columns.")
    st.stop()

source_rows = len(original)
source_columns = len(original.columns)
source_missing = int(original.isna().sum().sum())
source_duplicates = int(original.duplicated().sum())

data = original.drop_duplicates().copy()
duplicates_removed = source_rows - len(data)

numeric_columns = data.select_dtypes(include=np.number).columns.tolist()

# Convert columns that contain numeric values stored as text.
for col in data.columns:
    if col not in numeric_columns:
        converted = pd.to_numeric(data[col], errors="coerce")
        non_empty = data[col].notna()

        if non_empty.sum() > 0 and converted[non_empty].notna().all():
            data[col] = converted
            numeric_columns.append(col)

if len(numeric_columns) < 2:
    st.error("At least two numeric columns are required for clustering.")
    st.stop()

id_aliases = {"customerid", "id", "index", "rowid", "recordid"}
default_features = [
    col for col in numeric_columns if normalized(col) not in id_aliases
]
if len(default_features) < 2:
    default_features = numeric_columns.copy()


# =========================================================
# 7. SIDEBAR MODEL CONTROLS
# =========================================================
with st.sidebar:
    features = st.multiselect(
        "Clustering features",
        options=numeric_columns,
        default=default_features,
        help="Choose at least two numeric columns.",
    )

    if len(features) < 2:
        st.warning("Select at least two numeric features.")
        st.stop()

    selected_frame = data[features].replace([np.inf, -np.inf], np.nan)
    valid_mask = selected_frame.notna().all(axis=1)
    usable_count = int(valid_mask.sum())

    if usable_count < 3:
        st.error("At least three complete records are required.")
        st.stop()

    if selected_frame.loc[valid_mask].drop_duplicates().shape[0] < 3:
        st.error("The selected features have fewer than three unique records.")
        st.stop()

    max_allowed_k = min(10, usable_count - 1)

    max_k = st.slider(
        "Maximum K to compare",
        min_value=2,
        max_value=max_allowed_k,
        value=min(8, max_allowed_k),
    )

    automatic_k = st.toggle(
        "Automatic cluster selection",
        value=True,
        help="Uses the K with the highest silhouette score.",
    )

    manual_k = st.slider(
        "Manual cluster count",
        min_value=2,
        max_value=max_allowed_k,
        value=min(5, max_allowed_k),
        disabled=automatic_k,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("MODEL · K-MEANS")
    st.caption("SCALING · STANDARD SCALER")
    st.caption("EVALUATION · SILHOUETTE")


# =========================================================
# 8. PREPARE DATA AND RUN CLUSTERING
# =========================================================
valid_mask = selected_frame.notna().all(axis=1)
excluded_rows = int((~valid_mask).sum())

model_frame = selected_frame.loc[valid_mask].copy()
customer_data = data.loc[valid_mask].copy()

scaler = StandardScaler()
scaled = scaler.fit_transform(model_frame)

comparison = build_comparison(scaled, max_k)
valid_scores = comparison.dropna(subset=["Silhouette Score"])

if valid_scores.empty:
    st.error("Could not calculate a valid silhouette score.")
    st.stop()

best_row = valid_scores.loc[valid_scores["Silhouette Score"].idxmax()]
best_k = int(best_row["K"])
best_score = float(best_row["Silhouette Score"])

selected_k = best_k if automatic_k else min(manual_k, max_k)

model = KMeans(n_clusters=selected_k, random_state=42, n_init=10)
labels = model.fit_predict(scaled) + 1

customer_data = customer_data.copy()
customer_data["Cluster"] = labels

final_score = silhouette_score(scaled, labels - 1)

income_col = find_column(
    customer_data.columns,
    ["Annual Income (k$)", "Annual Income", "Income", "Customer Income"],
)
spending_col = find_column(
    customer_data.columns,
    ["Spending Score (1-100)", "Spending Score", "Spending", "Purchase Score"],
)
# Marketing labels are calculated from the model features, so only use these
# recognized fields when they are actually included in the current clustering.
if income_col not in features:
    income_col = None
if spending_col not in features:
    spending_col = None

income_median = customer_data[income_col].median() if income_col else None
spending_median = customer_data[spending_col].median() if spending_col else None

cluster_means = customer_data.groupby("Cluster")[features].mean()
cluster_counts = customer_data["Cluster"].value_counts().sort_index()

age_col = find_column(customer_data.columns, ["Age", "Customer Age"])
segment_names = build_segment_names(
    cluster_means,
    features,
    income_col,
    spending_col,
    income_median,
    spending_median,
    age_col=age_col,
)

customer_data["Segment"] = customer_data["Cluster"].map(segment_names)

# Build the summary in separate steps so feature names containing spaces,
# parentheses, or symbols are handled reliably across pandas versions.
summary = (
    customer_data.groupby(["Cluster", "Segment"])[features]
    .mean()
    .reset_index()
)
cluster_sizes = customer_data.groupby("Cluster").size()
summary["Customers"] = summary["Cluster"].map(cluster_sizes).astype(int)
summary["Customer Share (%)"] = (
    summary["Customers"] / len(customer_data) * 100
).round(1)
summary = summary.rename(
    columns={feature: f"Average {feature}" for feature in features}
)
summary = summary[
    ["Cluster", "Segment", "Customers", "Customer Share (%)"]
    + [f"Average {feature}" for feature in features]
]


# =========================================================
# 9. GLOBAL PAGE HEADER
# =========================================================
status = f"{len(customer_data):,} customers analyzed"

descriptions = {
    "Overview": (
        "Your customer data at a glance, with segment distribution and "
        "the most useful business signals."
    ),
    "Cluster Analysis": (
        "Evaluate model quality, compare cluster counts, and explore "
        "customer distributions in feature space."
    ),
    "Customer Explorer": (
        "Search individual records, filter customer groups, and export "
        "a focused customer list."
    ),
    "Reports & Data": (
        "Review dataset quality, cluster summaries, and downloadable "
        "analysis results."
    ),
    "Marketing Insights": (
        "Turn customer segments into practical campaign ideas, retention "
        "plays, and measurable marketing experiments."
    ),
}

show_header(page, descriptions[page], status)


# =========================================================
# 10. OVERVIEW
# =========================================================
if page == "Overview":

    # KPI row
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        show_metric(
            "CUSTOMERS",
            f"{len(customer_data):,}",
            f"{excluded_rows} excluded for selected-feature issues",
            "♙",
        )

    with k2:
        show_metric(
            "SEGMENTS",
            str(selected_k),
            "Active K-Means groups",
            "◉",
        )

    with k3:
        if income_col:
            income_value = customer_data[income_col].mean()
            show_metric(
                "AVG. INCOME",
                f"{income_value:,.1f}",
                income_col,
                "↗",
            )
        else:
            show_metric(
                "FEATURES",
                str(len(features)),
                "Features used by the model",
                "▦",
            )

    with k4:
        show_metric(
            "SILHOUETTE",
            f"{final_score:.3f}",
            "Higher often indicates clearer separation",
            "✓",
        )

    st.write("")

    # Main chart and segment distribution
    left, right = st.columns([1.65, 1])

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        show_section(
            "Customer segmentation map",
            "A two-feature view of customers, colored by cluster.",
        )

        x_feature = st.selectbox(
            "Horizontal axis",
            features,
            index=0,
            key="overview_x",
        )
        y_feature = st.selectbox(
            "Vertical axis",
            features,
            index=min(1, len(features) - 1),
            key="overview_y",
        )

        plot_data = customer_data.copy()
        st.altair_chart(
            scatter_chart(plot_data, x_feature, y_feature, height=350),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        show_section(
            "Segment distribution",
            "Customer counts by cluster.",
        )

        distribution = (
            customer_data["Cluster"]
            .value_counts()
            .sort_index()
            .rename_axis("Cluster")
            .reset_index(name="Customers")
        )
        distribution["Cluster"] = (
            "Cluster " + distribution["Cluster"].astype(str)
        )

        st.bar_chart(
            distribution.set_index("Cluster")[["Customers"]],
            height=350,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Segment cards
    section_title = "Customer segments"
    st.markdown(
        f'<div class="panel-heading">{section_title}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="panel-subtitle">A concise profile of each discovered group.</div>',
        unsafe_allow_html=True,
    )

    profile_cols = st.columns(3)

    for index, cid in enumerate(sorted(segment_names)):
        row = summary[summary["Cluster"] == cid].iloc[0]
        details = []

        for feature in features[:3]:
            details.append(
                f"{feature}: {row[f'Average {feature}']:.1f}"
            )

        with profile_cols[index % 3]:
            st.markdown(f"""
            <div class="segment-card">
                <div class="segment-label">SEGMENT {cid:02d}</div>
                <div class="segment-title">{segment_names[cid]}</div>
                <div class="segment-count">{int(row["Customers"]):,}</div>
                <div class="segment-caption">
                    customers · {row["Customer Share (%)"]:.1f}% share
                </div>
                <div class="segment-details">
                    {"<br>".join(details)}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Insights
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    show_section(
        "Business signals",
        "Use these patterns as starting points for further investigation.",
    )

    insight_list = []

    largest_cluster = int(cluster_counts.idxmax())
    largest_share = cluster_counts[largest_cluster] / len(customer_data) * 100
    insight_list.append(
        f"Cluster {largest_cluster} is the largest group, with "
        f"{cluster_counts[largest_cluster]} customers ({largest_share:.1f}%)."
    )

    if spending_col:
        high_spending = int(cluster_means[spending_col].idxmax())
        low_spending = int(cluster_means[spending_col].idxmin())

        insight_list.append(
            f"{segment_names[high_spending]} has the highest average "
            f"spending score ({cluster_means.loc[high_spending, spending_col]:.1f})."
        )
        insight_list.append(
            f"{segment_names[low_spending]} has the lowest average "
            f"spending score ({cluster_means.loc[low_spending, spending_col]:.1f}). "
            "Customer feedback may help explain the pattern."
        )

    if income_col:
        high_income = int(cluster_means[income_col].idxmax())
        insight_list.append(
            f"{segment_names[high_income]} has the highest average income "
            f"({cluster_means.loc[high_income, income_col]:.1f})."
        )

    for number, insight in enumerate(insight_list, start=1):
        st.markdown(
            f'<div class="insight-row"><span class="insight-number">'
            f'{number:02d}</span>{insight}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# 11. CLUSTER ANALYSIS
# =========================================================
elif page == "Cluster Analysis":

    k1, k2, k3 = st.columns(3)

    with k1:
        show_metric("BEST SILHOUETTE K", str(best_k),
                    "Highest tested silhouette score", "◉")
    with k2:
        show_metric("BEST SCORE", f"{best_score:.3f}",
                    "Among the tested K values", "✓")
    with k3:
        show_metric("ACTIVE K", str(selected_k),
                    "Current model configuration", "⚙")

    st.write("")

    left, right = st.columns(2)

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        show_section(
            "Elbow Method",
            "Look for a bend where adding clusters provides smaller reductions in inertia.",
        )
        st.line_chart(
            comparison.set_index("K")[["Inertia"]],
            height=330,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        show_section(
            "Silhouette comparison",
            "Higher values generally indicate better-separated clusters.",
        )
        st.line_chart(
            comparison.set_index("K")[["Silhouette Score"]],
            height=330,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    show_section(
        "Cluster visualization",
        "Select two features to inspect how customer groups are distributed.",
    )

    xcol, ycol = st.columns(2)

    with xcol:
        x_feature = st.selectbox(
            "X-axis feature",
            features,
            index=0,
            key="analysis_x",
        )

    with ycol:
        y_feature = st.selectbox(
            "Y-axis feature",
            features,
            index=min(1, len(features) - 1),
            key="analysis_y",
        )

    chart_data = customer_data.copy()
    st.altair_chart(
        scatter_chart(chart_data, x_feature, y_feature, height=470),
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Principal Component Analysis (PCA)"):
        pca = PCA(n_components=2)
        coords = pca.fit_transform(scaled)

        pca_frame = pd.DataFrame({
            "Principal Component 1": coords[:, 0],
            "Principal Component 2": coords[:, 1],
            "Cluster": customer_data["Cluster"].astype(str).to_numpy(),
        })

        pca_chart = alt.Chart(pca_frame).mark_circle(size=85, opacity=0.82).encode(
            x=alt.X("Principal Component 1:Q", title="Principal Component 1", scale=alt.Scale(zero=False)),
            y=alt.Y("Principal Component 2:Q", title="Principal Component 2", scale=alt.Scale(zero=False)),
            color=alt.Color("Cluster:N", title="Cluster"),
            tooltip=["Cluster", alt.Tooltip("Principal Component 1:Q", format=".2f"), alt.Tooltip("Principal Component 2:Q", format=".2f")],
        ).properties(height=430).interactive()
        st.altair_chart(pca_chart, use_container_width=True)

        explained = pca.explained_variance_ratio_.sum() * 100
        st.caption(
            f"The first two components explain {explained:.1f}% of the "
            "variance in the standardized selected features."
        )

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    show_section("Cluster evaluation data")
    st.dataframe(
        comparison.round(4),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download K comparison CSV",
        to_csv(comparison.round(6)),
        "customer_segmentation_k_comparison.csv",
        "text/csv",
    )
    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# 12. CUSTOMER EXPLORER
# =========================================================
elif page == "Customer Explorer":

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    show_section(
        "Find customers",
        "Filter by cluster and search across customer attributes.",
    )

    filter_col, search_col = st.columns([1, 1.5])

    with filter_col:
        available_clusters = sorted(customer_data["Cluster"].unique())

        chosen_clusters = st.multiselect(
            "Customer segments",
            available_clusters,
            default=available_clusters,
            format_func=lambda cid: f"Cluster {cid} · {segment_names[cid]}",
        )

    with search_col:
        query = st.text_input(
            "Search records",
            placeholder="Search customer ID, age, income, spending, or segment...",
        )

    filtered = customer_data[
        customer_data["Cluster"].isin(chosen_clusters)
    ].copy()

    if query.strip():
        query_text = query.strip().lower()

        match_mask = filtered.astype(str).apply(
            lambda col: col.str.lower().str.contains(
                query_text, regex=False, na=False
            )
        ).any(axis=1)

        filtered = filtered[match_mask]

    a, b = st.columns(2)

    with a:
        show_metric(
            "MATCHING RECORDS",
            f"{len(filtered):,}",
            f"Out of {len(customer_data):,} analyzed customers",
            "⌕",
        )

    with b:
        show_metric(
            "ACTIVE SEGMENTS",
            str(len(chosen_clusters)),
            "Selected in the filter",
            "◉",
        )

    st.write("")
    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True,
        height=500,
    )

    st.download_button(
        "Download filtered customers",
        to_csv(filtered),
        "filtered_customer_segments.csv",
        "text/csv",
    )
    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# 13. REPORTS & DATA
# =========================================================
elif page == "Reports & Data":

    # A concise executive snapshot makes the report page easier to scan.
    r1, r2, r3 = st.columns(3)
    with r1:
        show_metric("ACTIVE MODEL", "K-Means", "Standardized selected features", "◉")
    with r2:
        show_metric("CLUSTER COUNT", str(selected_k), "Groups in the current result", "▦")
    with r3:
        show_metric("SILHOUETTE SCORE", f"{final_score:.3f}", "Current model separation score", "✓")

    st.write("")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    show_section(
        "Data quality",
        "Review the uploaded dataset and the records used for clustering.",
    )

    a, b, c, d = st.columns(4)

    with a:
        show_metric("SOURCE ROWS", f"{source_rows:,}",
                    "Uploaded records", "▤")
    with b:
        show_metric("SOURCE COLUMNS", f"{source_columns:,}",
                    "Uploaded fields", "▦")
    with c:
        show_metric("MISSING CELLS", f"{source_missing:,}",
                    "Across the original dataset", "!")
    with d:
        show_metric("DUPLICATES REMOVED", f"{duplicates_removed:,}",
                    "Exact duplicate rows", "✓")

    if source_missing == 0 and excluded_rows == 0:
        st.success("No missing cells were found in the source data or selected model features.")
    elif excluded_rows:
        st.warning(
            f"{excluded_rows} record(s) were excluded because selected "
            "features contained missing or infinite values."
        )
    else:
        st.info("Some source columns contain missing values. Review the data preview.")

    with st.expander("Source data preview", expanded=True):
        st.dataframe(data.head(25), use_container_width=True, hide_index=True)

    with st.expander("Column quality report"):
        quality = pd.DataFrame({
            "Column": data.columns,
            "Data type": [str(data[col].dtype) for col in data.columns],
            "Non-null values": [int(data[col].notna().sum()) for col in data.columns],
            "Missing values": [int(data[col].isna().sum()) for col in data.columns],
        })
        st.dataframe(quality, use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    show_section(
        "Cluster summary",
        "Customer counts, segment descriptions, and average feature values.",
    )
    st.dataframe(
        summary.round(3),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download cluster summary",
        to_csv(summary.round(5)),
        "customer_cluster_summary.csv",
        "text/csv",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    show_section(
        "Export results",
        "Download the outputs for further reporting or analysis.",
    )

    d1, d2 = st.columns(2)

    with d1:
        st.download_button(
            "Download clustered customer data",
            to_csv(customer_data),
            "customer_segmentation_results.csv",
            "text/csv",
            use_container_width=True,
        )

    with d2:
        st.download_button(
            "Download K comparison data",
            to_csv(comparison.round(6)),
            "customer_segmentation_k_comparison.csv",
            "text/csv",
            use_container_width=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)



# =========================================================
# 14. MARKETING INSIGHTS
# =========================================================
elif page == "Marketing Insights":
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    show_section("Segment opportunity briefing", "Recommendations are derived from current cluster profiles. Treat them as hypotheses to test, not guaranteed outcomes.")
    total_customers = len(customer_data)
    largest_id = int(cluster_counts.idxmax())
    largest_share = float(cluster_counts[largest_id] / total_customers * 100)
    a, b, c, d = st.columns(4)
    with a:
        show_metric("CUSTOMERS REACHED", f"{total_customers:,}", "Records included in this analysis", "♙")
    with b:
        show_metric("SEGMENTS", str(selected_k), "Distinct groups in the active model", "◉")
    with c:
        show_metric("LARGEST SEGMENT", f"{largest_share:.1f}%", f"Cluster {largest_id} · {cluster_counts[largest_id]} customers", "↗")
    with d:
        show_metric("MODEL SILHOUETTE", f"{final_score:.3f}", "Use alongside business validation", "✓")
    st.markdown("</div>", unsafe_allow_html=True)

    marketing_rows = []
    for cid in sorted(segment_names):
        profile = cluster_means.loc[cid]
        count = int(cluster_counts[cid])
        share = count / total_customers * 100
        avg_income = float(profile[income_col]) if income_col else np.nan
        avg_spending = float(profile[spending_col]) if spending_col else np.nan
        avg_age = float(profile[age_col]) if age_col and age_col in features else (float(customer_data.loc[customer_data["Cluster"] == cid, age_col].mean()) if age_col else np.nan)

        if income_col and spending_col:
            if avg_income >= income_median and avg_spending < spending_median:
                opportunity, action, priority = "High-value conversion", "Test a personalized offer, product education, or a low-friction first-purchase incentive.", "High"
            elif avg_income >= income_median and avg_spending >= spending_median:
                opportunity, action, priority = "Premium loyalty", "Test VIP benefits, early access, bundles, or a referral reward; protect margin before discounting.", "High"
            elif avg_income < income_median and avg_spending >= spending_median:
                opportunity, action, priority = "Value-led engagement", "Promote affordable bundles, rewards, and complementary products rather than broad discounts.", "Medium"
            else:
                opportunity, action, priority = "Reactivation / discovery", "Test preference surveys, tailored recommendations, and a small re-engagement campaign.", "Medium"
        elif spending_col:
            if avg_spending >= spending_median:
                opportunity, action, priority = "Retention and advocacy", "Test loyalty benefits, referral prompts, and relevant cross-sell recommendations.", "High"
            else:
                opportunity, action, priority = "Engagement discovery", "Test product education, tailored recommendations, and feedback collection.", "Medium"
        else:
            opportunity, action, priority = "Learn and validate", "Interview a small sample and compare response rates across customer groups before targeting.", "Explore"

        marketing_rows.append({"Cluster": int(cid), "Segment": segment_names[cid], "Customers": count, "Share (%)": round(share, 1), "Average Income": round(avg_income, 1) if pd.notna(avg_income) else None, "Average Spending": round(avg_spending, 1) if pd.notna(avg_spending) else None, "Average Age": round(avg_age, 1) if pd.notna(avg_age) else None, "Opportunity": opportunity, "Priority": priority, "Suggested test": action})
    marketing_df = pd.DataFrame(marketing_rows)

    left, right = st.columns([1.2, 1])
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        show_section("Opportunity by segment", "Use segment size and behavior together to prioritize small, testable campaigns.")
        st.dataframe(marketing_df[["Cluster", "Segment", "Customers", "Share (%)", "Opportunity", "Priority"]], use_container_width=True, hide_index=True)
        st.download_button("Download marketing opportunity plan", to_csv(marketing_df), "nexus_marketing_opportunities.csv", "text/csv", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        show_section("Segment size", "Customer count by cluster")
        dist = marketing_df.set_index("Segment")[["Customers"]].sort_values("Customers", ascending=False)
        st.bar_chart(dist, height=330)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    show_section("Suggested campaign experiments", "Start with one measurable experiment per audience; compare against a holdout group where possible.")
    for _, row in marketing_df.sort_values(["Priority", "Customers"], ascending=[True, False]).iterrows():
        card = (f'<div class="recommendation"><div class="recommendation-title">Cluster {int(row["Cluster"]):02d} · {row["Opportunity"]} <span class="tag">{row["Priority"]} priority</span></div>'
                f'<div class="recommendation-copy"><b>{row["Segment"]}</b> · {int(row["Customers"])} customers ({row["Share (%)"]:.1f}% of analyzed records). {row["Suggested test"]}</div></div>')
        st.markdown(card, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    show_section("Measurement checklist", "Validate segmentation recommendations with real campaign results.")
    checks = ["Define one primary metric before launch: conversion, repeat purchase, response rate, or retention.", "Keep a control or holdout group where practical to estimate incremental impact.", "Compare outcomes by segment and campaign cost; income is not the same as profitability.", "Review segments periodically because customer behavior can change.", "Use appropriate, consented data and avoid sensitive or unfair targeting attributes."]
    for i, item in enumerate(checks, 1):
        st.markdown(f'<div class="insight-row"><span class="insight-number">{i:02d}</span>{item}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    show_section("Campaign scorecard template", "Illustrative starting targets only—set final thresholds from your own baseline and business economics.")
    scorecard = pd.DataFrame([
        {"Metric": "Conversion rate", "How to measure": "Conversions ÷ reached customers", "Initial test target": "+5% relative lift vs. control"},
        {"Metric": "Click / response rate", "How to measure": "Clicks or responses ÷ delivered messages", "Initial test target": "+10% relative lift vs. control"},
        {"Metric": "Average order value", "How to measure": "Revenue ÷ completed orders", "Initial test target": "No decline while conversion improves"},
        {"Metric": "Campaign ROI", "How to measure": "Incremental profit ÷ campaign cost", "Initial test target": "> 0 before scaling"},
    ])
    st.dataframe(scorecard, use_container_width=True, hide_index=True)
    st.caption("These are example experiment goals, not predictions or guaranteed results. Use a holdout group and compare against your baseline.")
    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# 15. FOOTER
# =========================================================
st.markdown("""
<div style="
    border-top:1px solid #2B3A51;
    margin-top:34px;
    padding-top:19px;
    display:flex;
    justify-content:space-between;
    gap:12px;
    flex-wrap:wrap;
    color:#8793A6;
    font-size:10px;
">
    <span><b style="color:#A9C9F8;">NEXUS</b> · CUSTOMER INTELLIGENCE</span>
    <span>Python · Streamlit · Pandas · Scikit-learn</span>
</div>
""", unsafe_allow_html=True)
