
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from preprocessing import load_and_clean_data, preprocess_for_clustering, SPEND_COLS
from clustering import (
    evaluate_k, apply_kmeans, generate_cluster_profiles,
    label_clusters, describe_segment, suggest_campaigns, recommended_k,
)
from market_basket import load_baskets, get_rules_for_cluster

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation · ML II",
    page_icon="🛒",
    layout="wide",
)

st.markdown("""
<style>
  .metric-card { background:#f0f4f8; border-radius:10px; padding:1rem 1.5rem; text-align:center; }
  .metric-card h2 { font-size:2rem; margin:0; color:#0d6efd; }
  .metric-card p  { margin:0; color:#6c757d; font-size:0.9rem; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")
info_path   = st.sidebar.text_input("customer_info path",   value="data/customer_info.csv")
basket_path = st.sidebar.text_input("customer_basket path", value="data/customer_basket.csv")
k_value     = st.sidebar.slider("Number of Clusters (K)", min_value=2, max_value=8, value=4)
run_rules   = st.sidebar.checkbox("Mine per-cluster association rules (slow)", value=False)


@st.cache_data(show_spinner="Loading and cleaning data…")
def get_data(ipath, bpath):
    return load_and_clean_data(ipath), load_baskets(bpath)

@st.cache_data(show_spinner="Evaluating K…")
def get_k_eval(X_arr):
    return evaluate_k(X_arr)

@st.cache_data(show_spinner="Clustering…")
def get_clusters(X_arr, k):
    return apply_kmeans(X_arr, n_clusters=k)


try:
    df_info, df_basket = get_data(info_path, basket_path)
except FileNotFoundError as e:
    st.error(f"File not found: {e}. Update the paths in the sidebar.")
    st.stop()

X_df, scaler, features = preprocess_for_clustering(df_info)
X_arr = X_df.values

clusters, pca_df, km_model = get_clusters(X_arr, k_value)
df_info["Cluster"]      = clusters
CLUSTER_NAMES           = label_clusters(df_info)   # data-driven, no hard-coding
df_info["Cluster_Name"] = df_info["Cluster"].map(CLUSTER_NAMES).fillna(df_info["Cluster"].astype(str))
pca_df["Cluster_Name"]  = df_info["Cluster_Name"].values

# CSV download
csv_bytes = df_info[["customer_id", "Cluster"]].to_csv(index=False).encode()
st.sidebar.download_button(
    "⬇️ Download customer_clusters.csv",
    data=csv_bytes,
    file_name="customer_clusters.csv",
    mime="text/csv",
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🛒 Customer Segmentation & Targeted Campaigns")
st.caption("Machine Learning II · NOVA IMS")
st.divider()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Customers",   f"{len(df_info):,}")
c2.metric("Clusters",          k_value)
c3.metric("Avg Total Spend",   f"€{df_info['total_spend'].mean():,.0f}")
c4.metric("Loyalty Card Rate", f"{df_info['has_loyalty_card'].mean()*100:.1f}%")
st.divider()

# ── Executive Summary ───────────────────────────────────────────────────────
st.markdown("### 📋 Executive Summary")
st.caption("Segments are discovered from **behavioural** spend patterns (K-Means on log-scaled "
           "category spend, promotion rate and product/store breadth). Demographics are used "
           "afterwards to profile and name each segment.")
for cid, cname in CLUSTER_NAMES.items():
    st.markdown(f"- **{cname}** — {describe_segment(df_info, cid)}")
st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA", "🗺️ Cluster Map", "👤 Cluster Profiles", "🎯 Campaigns"])

# ── TAB 1 · EDA ───────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Exploratory Data Analysis")
    col_a, col_b = st.columns(2)

    with col_a:
        fig = px.histogram(df_info, x="age", nbins=40,
                           color_discrete_sequence=["#0d6efd"],
                           title="Age Distribution", labels={"age": "Age (years)"})
        st.plotly_chart(fig, width="stretch")

    with col_b:
        spend_totals = df_info[SPEND_COLS].sum().sort_values(ascending=True)
        fig = px.bar(
            x=spend_totals.values,
            y=[c.replace("lifetime_spend_","").replace("_"," ").title() for c in spend_totals.index],
            orientation="h", title="Total Lifetime Spend by Category",
            labels={"x":"Total (€)","y":"Category"},
            color_discrete_sequence=["#0d6efd"],
        )
        st.plotly_chart(fig, width="stretch")

    col_c, col_d = st.columns(2)
    with col_c:
        fig = px.pie(df_info, names="customer_gender", title="Gender Split",
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig, width="stretch")
    with col_d:
        fig = px.bar(df_info["degree"].value_counts().reset_index(),
                     x="degree", y="count", title="Education Level",
                     labels={"degree":"Degree","count":"Customers"},
                     color_discrete_sequence=["#0d6efd"])
        st.plotly_chart(fig, width="stretch")

    st.markdown("### Optimal K Selection")
    k_eval = get_k_eval(X_arr)
    best_k = recommended_k(k_eval)
    if best_k == k_value:
        st.success(f"📈 Silhouette is highest at **K = {best_k}** — matching your current selection.")
    else:
        st.info(f"📈 Silhouette is highest at **K = {best_k}** "
                f"(you're currently viewing K = {k_value}; change it in the sidebar).")
    col_e, col_f = st.columns(2)
    with col_e:
        fig = px.line(k_eval.reset_index(), x="k", y="inertia", markers=True,
                      title="Elbow Curve (Inertia)")
        st.plotly_chart(fig, width="stretch")
    with col_f:
        fig = px.line(k_eval.reset_index(), x="k", y="silhouette", markers=True,
                      title="Silhouette Score", color_discrete_sequence=["#ff7043"])
        st.plotly_chart(fig, width="stretch")

# ── TAB 2 · CLUSTER MAP ───────────────────────────────────────────────────────
with tab2:
    st.markdown("### Customer Segments — PCA 2D Projection")
    fig = px.scatter(
        pca_df, x="PCA1", y="PCA2", color="Cluster_Name", opacity=0.55,
        title=f"K-Means Clusters (K={k_value}) — PCA 2D",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_traces(marker=dict(size=4))
    st.plotly_chart(fig, width="stretch")

    size_df = df_info["Cluster_Name"].value_counts().reset_index()
    size_df.columns = ["Segment", "Count"]
    fig = px.bar(size_df, x="Segment", y="Count", color="Segment",
                 color_discrete_sequence=px.colors.qualitative.Set2,
                 title="Number of Customers per Segment")
    st.plotly_chart(fig, width="stretch")

# ── TAB 3 · CLUSTER PROFILES ─────────────────────────────────────────────────
with tab3:
    st.markdown("### Average Spend per Category by Cluster")
    profile_df    = generate_cluster_profiles(df_info, cluster_col="Cluster")
    spend_profile = profile_df[[c for c in profile_df.columns if "lifetime_spend" in c]].copy()
    spend_profile.index   = spend_profile.index.map(CLUSTER_NAMES)
    spend_profile.columns = [c.replace("lifetime_spend_","").replace("_"," ").title()
                              for c in spend_profile.columns]

    fig = px.imshow(spend_profile, text_auto=".0f", color_continuous_scale="Blues",
                    title="Mean Spend (€) per Cluster per Category", aspect="auto")
    st.plotly_chart(fig, width="stretch")

    st.markdown("### Key Demographics per Cluster")
    demo_cols = ["age","total_kids","degree_level","years_customer",
                 "percentage_of_products_bought_promotion","total_spend"]
    demo = profile_df[demo_cols].copy()
    demo.index   = demo.index.map(CLUSTER_NAMES)
    demo.columns = ["Avg Age","Avg Kids","Degree Level","Years as Customer",
                    "% Promo Purchases","Avg Total Spend (€)"]
    st.dataframe(demo.style.background_gradient(cmap="Greens", axis=0), width="stretch")

    st.markdown("### Normalised Spending Radar")
    cats   = [c.replace("lifetime_spend_","").replace("_"," ").title() for c in SPEND_COLS]
    colors = px.colors.qualitative.Set2
    fig    = go.Figure()
    for i, (cid, cname) in enumerate(CLUSTER_NAMES.items()):
        row  = spend_profile.loc[cname].values.tolist() if cname in spend_profile.index else [0]*len(cats)
        norm = [v / (max(row) or 1) for v in row]
        fig.add_trace(go.Scatterpolar(
            r=norm+[norm[0]], theta=cats+[cats[0]], fill="toself",
            name=cname, line_color=colors[i], fillcolor=colors[i], opacity=0.3,
        ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                      title="Normalised Spend Radar")
    st.plotly_chart(fig, width="stretch")

# ── TAB 4 · CAMPAIGNS ────────────────────────────────────────────────────────
with tab4:
    st.markdown("### Targeted Marketing Strategies per Segment")
    if run_rules:
        st.info("Mining per-cluster association rules — this may take a few minutes…")

    for cid, cname in CLUSTER_NAMES.items():
        with st.expander(cname, expanded=True):
            seg    = df_info[df_info["Cluster"] == cid]
            m1, m2, m3 = st.columns(3)
            m1.metric("Customers",         f"{len(seg):,}")
            m2.metric("Avg Total Spend",   f"€{seg['total_spend'].mean():,.0f}")
            m3.metric("% Promo Purchases", f"{seg['percentage_of_products_bought_promotion'].mean()*100:.1f}%")

            st.markdown(f"_{describe_segment(df_info, cid)}_")

            st.markdown("**💡 Suggested Campaigns:**")
            for campaign in suggest_campaigns(df_info, cid):
                st.markdown(f"- {campaign}")

            if run_rules:
                rules_df = get_rules_for_cluster(df_info, df_basket, cid)
                if not rules_df.empty:
                    st.markdown("**🔗 Top Product Associations:**")
                    disp = rules_df[["antecedents","consequents","support","confidence","lift"]].copy()
                    disp["antecedents"] = disp["antecedents"].apply(lambda x: ", ".join(list(x)))
                    disp["consequents"] = disp["consequents"].apply(lambda x: ", ".join(list(x)))
                    st.dataframe(disp, width="stretch")