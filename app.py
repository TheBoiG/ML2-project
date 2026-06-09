import glob
import os
import re

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.colors import LinearSegmentedColormap

from preprocessing import load_and_clean_data, preprocess_for_clustering, SPEND_COLS
from clustering import (
    evaluate_k, apply_kmeans, generate_cluster_profiles,
    label_clusters, describe_segment, suggest_campaigns, recommended_k, project_pca,
)
from market_basket import load_baskets, get_rules_for_cluster

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation · ML II",
    layout="wide",
)

# ── Design tokens ─────────────────────────────────────────────────────────────
# Hybrid: Linear-style chrome (indigo accent, hairline borders, dense type) with
# soft pastel charts for readability. One palette drives every chart.
# Curated, teal-led segment palette (less "default rainbow", segment 0 ties to brand).
PALETTE = ["#3E9C8E", "#E8917E", "#E0B35E", "#7E8CC4",
           "#6FB7AD", "#D98CA8", "#A9B36B", "#8FA0B8"]
ACCENT      = "#3E9C8E"   # brand teal — single-series charts
ACCENT_WARM = "#E8917E"   # warm coral — contrast line
BRAND       = "#0F766E"   # deep teal — signature interactive accent (chrome + highlights)
BRAND_SOFT  = "#E1F0ED"   # tinted teal — badges / focus row
INDIGO      = BRAND       # alias kept for chart highlight references
SEQ_SCALE   = [[0.0, "#F1F8F6"], [0.5, "#9FD0C8"], [1.0, "#36897C"]]
INK   = "#3B3F46"   # primary text
MUTED = "#8A909B"   # axis ticks / secondary
GRID  = "#ECEDF1"   # gridlines / hairlines
DIMMED = "#D7DBE2"  # greyed-out (non-focused) segments

PASTEL_GREEN_CMAP = LinearSegmentedColormap.from_list(
    "pastel_green", ["#FBFDFB", "#DDEFE2", "#9ECFA8"])

FONT = "Nunito, 'Segoe UI', sans-serif"

ALL = "All segments"


def style_fig(fig, height=None, hovermode="closest", lock=False, value_axis=None):
    """Shared chart styling + smooth transitions. Call before every render.

    lock=True   → freeze both axes (no zoom/pan drift below the data).
    value_axis  → "y" or "x": pin that axis to start exactly at 0 (no negative pad).
    """
    fig.update_layout(
        font=dict(family=FONT, size=13, color=INK),
        title=dict(font=dict(family=FONT, size=15, color=INK),
                   x=0.01, xanchor="left", y=0.97, yanchor="top"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=PALETTE,
        margin=dict(l=12, r=12, t=52, b=12),
        barcornerradius=6,
        bargap=0.12,
        hovermode=hovermode,
        legend=dict(bgcolor="rgba(0,0,0,0)", title_text="",
                    font=dict(size=12, color=MUTED), orientation="h",
                    yanchor="bottom", y=1.02, xanchor="left", x=0),
        hoverlabel=dict(font=dict(family=FONT, size=13, color=INK),
                        bgcolor="white", bordercolor=GRID),
        coloraxis_colorbar=dict(outlinewidth=0, tickfont=dict(color=MUTED, size=11)),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=GRID, automargin=True,
                     tickfont=dict(color=MUTED, size=11), title_font=dict(color=MUTED, size=12))
    fig.update_yaxes(showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False,
                     linecolor="rgba(0,0,0,0)", automargin=True,
                     tickfont=dict(color=MUTED, size=11), title_font=dict(color=MUTED, size=12))
    if value_axis == "y":
        fig.update_yaxes(rangemode="tozero")
    elif value_axis == "x":
        fig.update_xaxes(rangemode="tozero")
    if lock:
        fig.update_xaxes(fixedrange=True)
        fig.update_yaxes(fixedrange=True)
    if height:
        fig.update_layout(height=height)
    return fig


# Full zoom only where the data is dense enough to need it (PCA scatter).
PLOT_CONFIG_ZOOM = {
    "displaylogo": False, "scrollZoom": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    "toImageButtonOptions": {"format": "png", "scale": 2},
}
# Categorical / distribution charts: kill zoom, keep only download.
PLOT_CONFIG_STATIC = {
    "displaylogo": False, "scrollZoom": False,
    "modeBarButtonsToRemove": ["zoom2d", "pan2d", "select2d", "lasso2d",
                               "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"],
    "toImageButtonOptions": {"format": "png", "scale": 2},
}


def show(fig, key, height=None, hovermode="closest", lock=False, value_axis=None, on_select="ignore"):
    cfg = PLOT_CONFIG_STATIC if lock else PLOT_CONFIG_ZOOM
    return st.plotly_chart(style_fig(fig, height, hovermode, lock, value_axis),
                           width="stretch", config=cfg, key=key, on_select=on_select)


# Client-side hover highlight: the hovered slice / bar / radar trace pops while the
# rest dim, animated via CSS transitions. Runs entirely in the chart's iframe — no
# Streamlit round-trip (Streamlit can't stream hover events back to the server).
_HOVER_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700&display=swap');
  body { margin:0; background:transparent; }
  .slice path, .point > path, .points path, .trace path, .surface {
    transition: opacity .22s ease, fill-opacity .22s ease, transform .22s ease;
  }
  .slice { transition: transform .22s ease; }
</style>
"""
_HOVER_JS = """
<script>
(function(){
  function init(){
    var gd=document.getElementById('hl');
    if(!gd||!gd.on||!gd.data){return setTimeout(init,120);}
    function arr(n,v){return Array.from({length:n},function(){return v;});}
    var ORIG=gd.data.map(function(t){
      return {
        opacity:(t.opacity==null?1:t.opacity),
        pull:(t.pull!=null?(Array.isArray(t.pull)?t.pull.slice():t.pull):null),
        mop:(t.marker&&t.marker.opacity!=null)?(Array.isArray(t.marker.opacity)?t.marker.opacity.slice():t.marker.opacity):null
      };
    });
    function restore(){
      gd.data.forEach(function(t,cn){
        if(t.type==='pie'){ if(ORIG[cn].pull!=null) Plotly.restyle(gd,{pull:[ORIG[cn].pull]},[cn]); }
        else if(t.type==='bar'){ Plotly.restyle(gd,{'marker.opacity':[ORIG[cn].mop!=null?ORIG[cn].mop:1]},[cn]); }
      });
      if(gd.data[0]&&gd.data[0].type==='scatterpolar'){
        var idx=gd.data.map(function(t,i){return i;});
        Plotly.restyle(gd,{opacity:idx.map(function(i){return ORIG[i].opacity;})},idx);
      }
    }
    gd.on('plotly_hover',function(ev){
      var p=ev.points[0], cn=p.curveNumber, pn=p.pointNumber, tr=gd.data[cn];
      if(tr.type==='pie'){
        var n=tr.labels.length, pull=arr(n,0.0); pull[pn]=0.16;
        Plotly.restyle(gd,{pull:[pull]},[cn]);
      } else if(tr.type==='bar'){
        var n=(tr.y&&tr.y.length)||tr.x.length, op=arr(n,0.22); op[pn]=1.0;
        Plotly.restyle(gd,{'marker.opacity':[op]},[cn]);
      } else if(tr.type==='scatterpolar'){
        var idx=gd.data.map(function(t,i){return i;});
        Plotly.restyle(gd,{opacity:idx.map(function(i){return i===cn?0.9:0.05;})},idx);
      }
    });
    gd.on('plotly_unhover',restore);
  }
  init();
})();
</script>
"""


def interactive_chart(fig, height, zoom=False):
    """Render a figure with animated hover-highlight (rich tooltips).
    zoom=True keeps scroll/drag zoom (used for the radar)."""
    cfg = {"displaylogo": False, "responsive": True}
    if zoom:
        cfg.update({"scrollZoom": True, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})
    else:
        cfg["displayModeBar"] = False
    inner = fig.to_html(include_plotlyjs="cdn", full_html=False, div_id="hl", config=cfg)
    components.html(_HOVER_CSS + inner + _HOVER_JS, height=height + 16, scrolling=False)


def ihover(fig, height, lock=True, value_axis=None, hovermode="closest"):
    interactive_chart(style_fig(fig, height, hovermode, lock, value_axis), height)


# Donut: the hovered slice brightens + pulls out while the rest fade to grey.
_DONUT_JS = """
<script>
(function(){
  function init(){
    var gd=document.getElementById('hl');
    if(!gd||!gd.on||!gd.data||!gd.data[0]){return setTimeout(init,120);}
    var BASE=(gd.data[0].marker&&gd.data[0].marker.colors)?gd.data[0].marker.colors.slice():null;
    if(!BASE){return;}
    var DIM='#E4E7EB';
    gd.on('plotly_hover',function(ev){
      var pn=ev.points[0].pointNumber;
      var pull=BASE.map(function(_,i){return i===pn?0.10:0.0;});
      var cols=BASE.map(function(c,i){return i===pn?c:DIM;});
      Plotly.restyle(gd,{pull:[pull],'marker.colors':[cols]},[0]);
    });
    gd.on('plotly_unhover',function(){
      Plotly.restyle(gd,{pull:[BASE.map(function(){return 0;})],'marker.colors':[BASE]},[0]);
    });
  }
  init();
})();
</script>
"""


def donut_chart(fig, height):
    """Donut with a brighten/dim hover highlight."""
    style_fig(fig, height, lock=True)
    inner = fig.to_html(include_plotlyjs="cdn", full_html=False, div_id="hl",
                        config={"displaylogo": False, "responsive": True, "displayModeBar": False})
    components.html(_HOVER_CSS + inner + _DONUT_JS, height=height + 16, scrolling=False)


# Radar: wheel zooms the radial axis toward the centre (so the squished small
# categories open up); the current [0,1] scale is the furthest zoom-out. The
# hover-highlight from _HOVER_JS (scatterpolar branch) runs alongside it.
_RADAR_JS = """
<script>
(function(){
  function init(){
    var gd=document.getElementById('hl');
    if(!gd||!gd.on||!window.Plotly){return setTimeout(init,120);}
    var rmax=1;  // current = furthest zoom-out
    gd.addEventListener('wheel',function(e){
      e.preventDefault();
      rmax+=(e.deltaY<0?-0.12:0.12);
      if(rmax>1)rmax=1; if(rmax<0.2)rmax=0.2;
      Plotly.relayout(gd,{'polar.radialaxis.range':[0,rmax]});
    },{passive:false});
    gd.addEventListener('dblclick',function(){ rmax=1; Plotly.relayout(gd,{'polar.radialaxis.range':[0,1]}); });
  }
  init();
})();
</script>
"""


def radar_chart(fig, height):
    """Radar with hover-highlight + capped wheel zoom into the centre."""
    inner = fig.to_html(include_plotlyjs="cdn", full_html=False, div_id="hl",
                        config={"displaylogo": False, "responsive": True, "displayModeBar": False})
    components.html(_HOVER_CSS + inner + _HOVER_JS + _RADAR_JS, height=height + 16, scrolling=False)


# ── Global look: round-readable font + Linear-style chrome ────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

  html, body, [class*="css"], .stApp, .stMarkdown, button, input, select, textarea,
  [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
      font-family: 'Nunito', 'Segoe UI', sans-serif !important;
  }
  h1, h2, h3, h4 {
      font-family: 'Space Grotesk', 'Nunito', sans-serif !important;
      font-weight: 600 !important; color: #20303A; letter-spacing: -0.02em;
  }
  /* Signature: teal accent rule before each section heading */
  [data-testid="stMarkdownContainer"] h3 { position:relative; padding-left:15px; }
  [data-testid="stMarkdownContainer"] h3::before {
      content:''; position:absolute; left:0; top:.16em; bottom:.16em;
      width:4px; border-radius:3px; background:#0F766E;
  }
  .stApp { background: #FBFBFC; }
  .block-container { padding-top: 2.2rem; }

  @keyframes fadeUp { from {opacity:0; transform:translateY(4px);} to {opacity:1; transform:none;} }
  [data-testid="stVerticalBlockBorderWrapper"] { animation: fadeUp .35s ease both; }

  [data-testid="stMetric"] {
      background:#FFFFFF; border:1px solid #ECECEF; border-radius:12px;
      padding:0.85rem 1.1rem; box-shadow:0 1px 2px rgba(20,22,28,0.04);
  }
  [data-testid="stMetricValue"] { color:#2B2F36; font-weight:800; font-size:1.8rem; }
  [data-testid="stMetricLabel"] { color:#8A909B; font-weight:600;
      text-transform:uppercase; letter-spacing:0.04em; font-size:0.72rem; }

  [data-testid="stVerticalBlockBorderWrapper"] {
      border-radius:14px; border-color:#ECECEF !important;
  }

  .stTabs [data-baseweb="tab-list"] { gap:6px; border-bottom:none; }
  .stTabs [data-baseweb="tab"] {
      border-radius:10px; padding:7px 18px; background:#F2F2F4;
      font-weight:600; color:#6B7079; transition:all .15s ease;
  }
  .stTabs [data-baseweb="tab"]:hover { background:#E9E9EE; color:#3B3F46; }
  .stTabs [aria-selected="true"] { background:#0F766E !important; color:#FFFFFF !important; }
  .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none; }

  [data-testid="stSidebar"] { background:#F6F7F9; border-right:1px solid #ECECEF; }
  [data-testid="stSidebar"] h1 { font-size:1.3rem; letter-spacing:-0.01em; }
  [data-testid="stSidebar"] h5 {
      text-transform:uppercase; letter-spacing:0.08em; font-size:0.7rem;
      color:#9AA0AB !important; font-weight:700 !important; margin:1.1rem 0 0.2rem 0;
  }

  .stButton button, .stDownloadButton button {
      border-radius:10px; font-weight:700; border:1px solid #DEDFE6;
  }
  .stDownloadButton button:hover { border-color:#0F766E; color:#0F766E; }
  [data-baseweb="select"] > div { border-radius:10px; }

  [data-testid="stExpander"] { border:1px solid #ECECEF; border-radius:14px; }

  /* Right-align the 2D/3D projection toggle flush to its column's right edge */
  [data-testid="stElementContainer"]:has([data-testid="stButtonGroup"]) {
      align-self:flex-end !important;
  }
  [data-testid="stButtonGroup"] { justify-content:flex-end !important; }

  hr { margin:0.6rem 0; border-color:#ECECEF; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar · Data + Model (choose, don't type) ───────────────────────────────
def discover_csvs():
    # Canonical course data lives in data/ — list it first so it stays the
    # default even if look-alike files exist elsewhere (e.g. an older DATA/ copy).
    primary = sorted(glob.glob("data/*.csv"))
    extra = sorted(set(glob.glob("DATA/**/*.csv", recursive=True) + glob.glob("*.csv")) - set(primary))
    return primary + extra


def default_index(paths, needle):
    for i, p in enumerate(paths):
        if needle in os.path.basename(p).lower():
            return i
    return 0


csv_paths = discover_csvs()

with st.sidebar:
    st.title("Controls")
    st.caption("Customer Segmentation · ML II · NOVA IMS")

    st.markdown("##### Data")
    if csv_paths:
        info_path = st.selectbox("Customer info dataset", csv_paths,
                                 index=default_index(csv_paths, "info"),
                                 format_func=os.path.basename)
        basket_path = st.selectbox("Customer basket dataset", csv_paths,
                                   index=default_index(csv_paths, "basket"),
                                   format_func=os.path.basename)
    else:
        st.warning("No CSV files found under data/ — using defaults.")
        info_path, basket_path = "data/customer_info.csv", "data/customer_basket.csv"

    st.markdown("##### Model")
    k_value = st.slider("Number of clusters (K)", min_value=2, max_value=8, value=4)
    run_rules = st.toggle("Mine per-cluster association rules", value=False,
                          help="Slower — mines Apriori product associations for each segment.")


@st.cache_data(show_spinner="Loading and cleaning data…")
def get_data(ipath, bpath):
    return load_and_clean_data(ipath), load_baskets(bpath)

@st.cache_data(show_spinner="Evaluating K…")
def get_k_eval(X_arr):
    return evaluate_k(X_arr)

@st.cache_data(show_spinner="Clustering…")
def get_clusters(X_arr, k):
    return apply_kmeans(X_arr, n_clusters=k)

@st.cache_data(show_spinner="Projecting…")
def get_pca(X_arr, n):
    return project_pca(X_arr, n_components=n)


try:
    df_info, df_basket = get_data(info_path, basket_path)
except FileNotFoundError as e:
    st.error(f"File not found: {e}. Pick a different dataset in the sidebar.")
    st.stop()

X_df, scaler, features = preprocess_for_clustering(df_info)
X_arr = X_df.values

clusters, pca_df, km_model = get_clusters(X_arr, k_value)
df_info["Cluster"]      = clusters
CLUSTER_NAMES           = label_clusters(df_info)   # data-driven, no hard-coding
df_info["Cluster_Name"] = df_info["Cluster"].map(CLUSTER_NAMES).fillna(df_info["Cluster"].astype(str))
pca_df["Cluster_Name"]  = df_info["Cluster_Name"].values

SEG_LIST   = list(CLUSTER_NAMES.values())
SEG_COLORS = {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(SEG_LIST)}
FOCUS_OPTIONS = [ALL] + SEG_LIST

# ── Sidebar · Focus (cross-chart highlight) + Export ──────────────────────────
# A click on a chart (handled later) lands here as _click_focus and is applied
# BEFORE the focus widget is built, so chart-clicks and the box stay in sync.
if st.session_state.get("focus_box") not in FOCUS_OPTIONS:
    st.session_state["focus_box"] = ALL
if "_click_focus" in st.session_state:
    cf = st.session_state.pop("_click_focus")
    if cf in FOCUS_OPTIONS:
        st.session_state["focus_box"] = cf

with st.sidebar:
    st.markdown("##### Focus")
    focus = st.selectbox("Highlight a segment", FOCUS_OPTIONS, key="focus_box",
                         help="Dims every other segment across all charts and filters "
                              "the EDA tab. Tip: you can also click a bar or cluster point.")
    if st.button("⟲  Reset focus", width="stretch", disabled=(focus == ALL)):
        st.session_state["_click_focus"] = ALL
        st.rerun()

    st.markdown("##### Export")
    csv_bytes = df_info[["customer_id", "Cluster"]].to_csv(index=False).encode()
    st.download_button("Download customer_clusters.csv", data=csv_bytes,
                       file_name="customer_clusters.csv", mime="text/csv",
                       width="stretch")


def seg_color(name):
    return SEG_COLORS.get(name, ACCENT) if focus in (ALL, name) else DIMMED


def seg_alpha(name, hi=1.0, lo=0.12):
    return hi if focus in (ALL, name) else lo


def click_to_focus(event, key, mapper):
    """Turn a chart point-click into a focus change, gated by selection signature
    so a stale selection never fights a manual focus pick."""
    pts = []
    try:
        pts = event["selection"]["points"]
    except (TypeError, KeyError):
        pts = []
    sig = tuple((p.get("point_index"), p.get("point_number"), p.get("curve_number")) for p in pts)
    store = st.session_state.setdefault("_sel_sigs", {})
    if store.get(key) == sig:
        return
    store[key] = sig
    if not pts:
        return
    seg = mapper(pts[0])
    if seg and seg in SEG_LIST and seg != focus:
        st.session_state["_click_focus"] = seg
        st.rerun()


# ── Header ────────────────────────────────────────────────────────────────────
st.title("Customer Segmentation & Targeted Campaigns")
st.caption("Machine Learning II · NOVA IMS"
           + ("" if focus == ALL else f"  ·  focused on **{focus}**"))
st.divider()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Customers",   f"{len(df_info):,}")
c2.metric("Clusters",          k_value)
c3.metric("Avg Total Spend",   f"€{df_info['total_spend'].mean():,.0f}")
c4.metric("Loyalty Card Rate", f"{df_info['has_loyalty_card'].mean()*100:.1f}%")
st.divider()

# ── Executive Summary ───────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown("### Executive Summary")
    st.caption("Segments are discovered from **behavioural** spend patterns (K-Means on log-scaled "
               "category spend, promotion rate and product/store breadth). Demographics are used "
               "afterwards to profile and name each segment.")
    for cid, cname in CLUSTER_NAMES.items():
        prefix = "▶ " if focus == cname else "- "
        line = f"{prefix}**{cname}** — {describe_segment(df_info, cid)}"
        if focus in (ALL, cname):
            st.markdown(line)
        else:
            st.markdown(f"<span style='opacity:0.4'>{line}</span>", unsafe_allow_html=True)

st.write("")
tab1, tab2, tab3, tab4 = st.tabs(["EDA", "Cluster Map", "Cluster Profiles", "Campaigns"])

# ── TAB 1 · EDA (reacts to focus) ─────────────────────────────────────────────
with tab1:
    eda_df = df_info if focus == ALL else df_info[df_info["Cluster_Name"] == focus]
    label  = "all customers" if focus == ALL else f"{focus} ({len(eda_df):,})"
    st.markdown("### Exploratory Data Analysis")
    st.caption(f"Showing: **{label}**")

    st.caption("Hover any bar, slice or radar trace to spotlight it.")
    col_a, col_b = st.columns(2)
    with col_a:
        with st.container(border=True):
            # A smooth density line suits a distribution better than 40 bars; the
            # y-axis is zoomed to the data band (not 0) so the shape is readable.
            counts, edges = np.histogram(eda_df["age"].dropna(), bins=40)
            centers = (edges[:-1] + edges[1:]) / 2
            lo, hi = counts.min(), counts.max()
            pad = max(1, (hi - lo)) * 0.4
            fig = go.Figure(go.Scatter(
                x=centers, y=counts, mode="lines", fill="tozeroy", line_shape="spline",
                line=dict(color="#4C86B6", width=2.5), fillcolor="rgba(76,134,182,0.15)",
                hovertemplate="<b>Age ~%{x:.0f}</b><br>%{y:,} customers<extra></extra>"))
            fig.update_layout(title="Age Distribution")
            fig.update_xaxes(title="Age (years)")
            fig.update_yaxes(title="count", range=[max(0, lo - pad), hi + pad * 0.5])
            ihover(fig, height=320)
    with col_b:
        with st.container(border=True):
            # One category (Groceries) dwarfs the rest — a treemap keeps every
            # category readable as an area-proportional, labelled tile.
            sp = eda_df[SPEND_COLS].sum().sort_values(ascending=False)
            tdf = pd.DataFrame({
                "Category": [c.replace("lifetime_spend_", "").replace("_", " ").title() for c in sp.index],
                "Spend": sp.values})
            fig = px.treemap(tdf, path=["Category"], values="Spend",
                             color="Category", color_discrete_sequence=PALETTE,
                             title="Total Lifetime Spend by Category")
            fig.update_traces(
                texttemplate="%{label}", textposition="middle center",
                textfont=dict(family=FONT, size=15, color="#1F2D36"),
                insidetextfont=dict(family=FONT, size=15, color="#1F2D36"),
                hovertemplate="<b>%{label}</b><br>€%{value:,.0f}<br>%{percentRoot} of spend<extra></extra>",
                marker=dict(line=dict(color="white", width=2)), root_color="white",
                tiling=dict(pad=2))
            fig.update_layout(uniformtext=dict(minsize=11, mode="hide"))
            ihover(fig, height=320)

    col_c, col_d = st.columns(2)
    with col_c:
        with st.container(border=True):
            # Pre-aggregate (2 rows) — keeps the payload tiny and lets the JS
            # read/dim explicit per-slice colours.
            gc = eda_df["customer_gender"].value_counts()
            fig = go.Figure(go.Pie(
                labels=gc.index.tolist(), values=gc.values.tolist(), hole=0.55, sort=False,
                marker=dict(colors=PALETTE[:len(gc)], line=dict(color="white", width=2)),
                textinfo="percent+label", textfont_size=13,
                hovertemplate="<b>%{label}</b><br>%{value:,} customers<br>%{percent}<extra></extra>"))
            fig.update_layout(title="Gender Split")
            donut_chart(fig, height=320)
    with col_d:
        with st.container(border=True):
            # Sorted horizontal bars + value labels — readable even when one
            # category (e.g. "None") dwarfs the rest.
            edu = eda_df["degree"].value_counts().sort_values(ascending=True).reset_index()
            edu.columns = ["degree", "count"]
            fig = go.Figure(go.Bar(
                x=edu["count"], y=edu["degree"], orientation="h",
                marker_color=[PALETTE[i % len(PALETTE)] for i in range(len(edu))],
                text=edu["count"], texttemplate="%{text:,}", textposition="outside", cliponaxis=False,
                hovertemplate="<b>%{y}</b><br>%{x:,} customers<extra></extra>"))
            fig.update_layout(title="Education Level")
            fig.update_xaxes(range=[0, edu["count"].max() * 1.18])
            ihover(fig, height=320)

    st.markdown("### Optimal K Selection")
    st.caption("Computed over **all** customers, independent of the focus filter.")
    k_eval = get_k_eval(X_arr)
    best_k = recommended_k(k_eval)
    if best_k == k_value:
        st.success(f"Silhouette is highest at **K = {best_k}** — matching your current selection.")
    else:
        st.info(f"Silhouette is highest at **K = {best_k}** "
                f"(you're currently viewing K = {k_value}; change it in the sidebar).")
    col_e, col_f = st.columns(2)
    with col_e:
        with st.container(border=True):
            fig = px.line(k_eval.reset_index(), x="k", y="inertia", markers=True,
                          title="Elbow Curve (Inertia)", color_discrete_sequence=[ACCENT])
            fig.update_traces(line=dict(width=3), marker=dict(size=9),
                              hovertemplate="<b>K = %{x}</b><br>Inertia %{y:,.0f}<extra></extra>")
            show(fig, key="k_elbow", height=320, hovermode="x unified", lock=True)
    with col_f:
        with st.container(border=True):
            fig = px.line(k_eval.reset_index(), x="k", y="silhouette", markers=True,
                          title="Silhouette Score", color_discrete_sequence=[ACCENT_WARM])
            fig.update_traces(line=dict(width=3), marker=dict(size=9),
                              hovertemplate="<b>K = %{x}</b><br>Silhouette %{y:.3f}<extra></extra>")
            show(fig, key="k_sil", height=320, hovermode="x unified", lock=True)

# ── TAB 2 · CLUSTER MAP (focus dims others; click a point/bar to focus) ───────
with tab2:
    head_l, head_r = st.columns([6, 1])
    head_l.markdown("### Customer Segments — PCA Projection")
    proj = head_r.segmented_control("Projection space", ["2D", "3D"], default="2D",
                                    selection_mode="single", key="proj_space",
                                    label_visibility="collapsed")
    proj = proj or "2D"

    n_comp = 3 if proj == "3D" else 2
    pca_df = get_pca(X_arr, n_comp)
    pca_df["Cluster_Name"] = df_info["Cluster_Name"].values

    with st.container(border=True):
        if proj == "2D":
            st.caption("Drag to zoom · scroll to scale · **click a point** to focus its segment.")
            fig = px.scatter(
                pca_df, x="PCA1", y="PCA2", color="Cluster_Name", opacity=0.6,
                title=f"K-Means Clusters (K={k_value}) — PCA 2D",
                color_discrete_map=SEG_COLORS, labels={"Cluster_Name": "Segment"},
                category_orders={"Cluster_Name": SEG_LIST},
            )
            for tr in fig.data:
                tr.update(marker=dict(size=5, line=dict(width=0)),
                          opacity=seg_alpha(tr.name, 0.7, 0.08),
                          hovertemplate="<b>%{fullData.name}</b>"
                                        "<br>PCA1 %{x:.2f} · PCA2 %{y:.2f}<extra></extra>")
            fig.update_layout(dragmode="zoom")
            ev = show(fig, key="map_pca", height=700, on_select="rerun")
            click_to_focus(ev, "map_pca",
                           lambda p: SEG_LIST[p["curve_number"]]
                           if p.get("curve_number") is not None and p["curve_number"] < len(SEG_LIST)
                           else None)
        else:
            st.caption("Drag to rotate · scroll to zoom · pick a segment in the sidebar to highlight.")
            fig = px.scatter_3d(
                pca_df, x="PCA1", y="PCA2", z="PCA3", color="Cluster_Name", opacity=0.6,
                title=f"K-Means Clusters (K={k_value}) — PCA 3D",
                color_discrete_map=SEG_COLORS, labels={"Cluster_Name": "Segment"},
                category_orders={"Cluster_Name": SEG_LIST},
            )
            for tr in fig.data:
                tr.update(marker=dict(size=3, line=dict(width=0)),
                          opacity=seg_alpha(tr.name, 0.8, 0.06),
                          hovertemplate="<b>%{fullData.name}</b>"
                                        "<br>PCA1 %{x:.2f} · PCA2 %{y:.2f} · PCA3 %{z:.2f}<extra></extra>")
            axis3d = dict(backgroundcolor="rgba(0,0,0,0)", showbackground=True, gridcolor=GRID,
                          zerolinecolor=GRID, color=MUTED)
            fig.update_layout(
                font=dict(family=FONT, size=13, color=INK),
                title=dict(font=dict(family=FONT, size=15, color=INK), x=0.01, xanchor="left"),
                paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=46, b=0), height=880,
                legend=dict(bgcolor="rgba(0,0,0,0)", title_text="", font=dict(size=12, color=MUTED),
                            orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
                scene=dict(bgcolor="rgba(0,0,0,0)", xaxis=axis3d, yaxis=axis3d, zaxis=axis3d),
            )
            st.plotly_chart(fig, width="stretch", key="map_pca3d",
                            config={"displaylogo": False, "scrollZoom": True,
                                    "modeBarButtonsToRemove": ["lasso2d", "select2d"]})

    with st.container(border=True):
        st.caption("Click a bar to focus that segment.")
        size_df = (df_info["Cluster_Name"].value_counts().reindex(SEG_LIST).reset_index())
        size_df.columns = ["Segment", "Count"]
        size_df = size_df.sort_values("Count")  # ascending → largest on top (horizontal)
        fig = go.Figure(go.Bar(
            x=size_df["Count"], y=size_df["Segment"], orientation="h",
            marker_color=[seg_color(s) for s in size_df["Segment"]],
            text=size_df["Count"], texttemplate="%{text:,}", textposition="outside", cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>%{x:,} customers<extra></extra>",
        ))
        fig.update_layout(title="Number of Customers per Segment", showlegend=False)
        fig.update_xaxes(range=[0, size_df["Count"].max() * 1.15])
        # Height scales with the number of segments so bars keep a constant
        # thickness instead of squashing at higher K.
        ev = show(fig, key="map_sizes", height=90 + 56 * len(size_df), lock=True, on_select="rerun")
        click_to_focus(ev, "map_sizes", lambda p: p.get("y"))

# ── TAB 3 · CLUSTER PROFILES ─────────────────────────────────────────────────
with tab3:
    st.markdown("### Average Spend per Category by Cluster")
    profile_df    = generate_cluster_profiles(df_info, cluster_col="Cluster")
    spend_profile = profile_df[[c for c in profile_df.columns if "lifetime_spend" in c]].copy()
    spend_profile.index   = spend_profile.index.map(CLUSTER_NAMES)
    spend_profile.columns = [c.replace("lifetime_spend_", "").replace("_", " ").title()
                             for c in spend_profile.columns]

    with st.container(border=True):
        fig = px.imshow(spend_profile, text_auto=".0f", color_continuous_scale=SEQ_SCALE,
                        title="Mean Spend (€) per Cluster per Category", aspect="auto")
        fig.update_traces(textfont=dict(family=FONT, size=11),
                          hovertemplate="<b>%{y}</b><br>%{x}: €%{z:,.0f}<extra></extra>")
        if focus != ALL and focus in list(spend_profile.index):
            r = list(spend_profile.index).index(focus)
            fig.add_shape(type="rect", x0=-0.5, x1=len(spend_profile.columns) - 0.5,
                          y0=r - 0.5, y1=r + 0.5, line=dict(color=INDIGO, width=2.5))
        fig.update_xaxes(side="bottom")
        show(fig, key="prof_heat", height=380, lock=True)

    st.markdown("### Key Demographics per Cluster")
    demo_cols = ["age", "total_kids", "degree_level", "years_customer",
                 "percentage_of_products_bought_promotion", "total_spend"]
    demo = profile_df[demo_cols].copy()
    demo.index   = demo.index.map(CLUSTER_NAMES)
    demo.columns = ["Avg Age", "Avg Kids", "Degree Level", "Years as Customer",
                    "% Promo Purchases", "Avg Total Spend (€)"]
    sty = demo.style.background_gradient(cmap=PASTEL_GREEN_CMAP, axis=0)
    if focus != ALL and focus in list(demo.index):
        sty = sty.apply(lambda row: ["background-color:#E1F0ED; font-weight:700"
                                     if row.name == focus else "" for _ in row], axis=1)
    st.dataframe(sty, width="stretch")

    st.markdown("### Normalised Spending Radar")
    cats = [c.replace("lifetime_spend_", "").replace("_", " ").title() for c in SPEND_COLS]
    fig  = go.Figure()
    for cid, cname in CLUSTER_NAMES.items():
        row  = spend_profile.loc[cname].values.tolist() if cname in spend_profile.index else [0] * len(cats)
        norm = [v / (max(row) or 1) for v in row]
        color = SEG_COLORS.get(cname, ACCENT)
        focused = focus in (ALL, cname)
        fig.add_trace(go.Scatterpolar(
            r=norm + [norm[0]], theta=cats + [cats[0]], fill="toself",
            name=cname.replace(" Shoppers", ""),   # short label; full name stays in the tooltip
            line=dict(color=color if focused else DIMMED, width=3 if focus == cname else 2),
            fillcolor=color, opacity=(0.45 if focus == cname else 0.28) if focused else 0.06,
            hovertemplate="%{theta}: %{r:.2f}<extra>" + cname + "</extra>",
        ))
    st.caption("Hover a segment's outline to bring it forward · scroll to zoom into the centre "
               "(the current scale is the furthest zoom-out) · double-click to reset.")
    style_fig(fig, height=600)
    fig.update_layout(
        title=dict(text="Normalised Spend Radar", y=0.99, yanchor="top"),
        transition=dict(duration=400, easing="cubic-in-out"),
        margin=dict(l=40, r=40, t=56, b=30),
        showlegend=False,   # replaced by a custom wrapping HTML legend below
        polar=dict(domain=dict(x=[0.06, 0.94], y=[0.0, 0.97]), bgcolor="rgba(0,0,0,0)",
                   radialaxis=dict(visible=True, range=[0, 1], minallowed=0, maxallowed=1,
                                   gridcolor=GRID, tickfont=dict(color=MUTED, size=10), linecolor=GRID),
                   angularaxis=dict(gridcolor=GRID, tickfont=dict(color=INK, size=11), linecolor=GRID)),
    )
    radar_chart(fig, height=600)
    chips = "".join(
        f"<span style='display:inline-flex;align-items:center;gap:7px;margin:0 16px 6px 0;"
        f"font-size:.86rem;color:#6B7079'><span style='width:11px;height:11px;border-radius:3px;"
        f"background:{SEG_COLORS[name]};display:inline-block;flex:none'></span>{name}</span>"
        for name in SEG_LIST)
    st.markdown(f"<div style='display:flex;flex-wrap:wrap;justify-content:center;margin-top:-6px'>"
                f"{chips}</div>", unsafe_allow_html=True)

# ── TAB 4 · CAMPAIGNS ────────────────────────────────────────────────────────
with tab4:
    st.markdown("### Targeted Marketing Strategies per Segment")
    st.caption("Every campaign is generated from the segment's own behavioural signature — "
               "its over-indexed categories, promotion appetite and family profile.")
    if run_rules:
        st.info("Mining per-cluster association rules — this may take a few minutes…")

    total_rev = df_info["total_spend"].sum()

    for cid, cname in CLUSTER_NAMES.items():
        seg     = df_info[df_info["Cluster"] == cid]
        color   = SEG_COLORS.get(cname, ACCENT)
        share   = len(seg) / len(df_info) * 100
        revenue = seg["total_spend"].sum()
        rev_pct = revenue / total_rev * 100
        promo   = seg["percentage_of_products_bought_promotion"].mean() * 100

        with st.container(border=True):
            badge = ("<span style='background:#E1F0ED;color:#0F766E;font-size:.68rem;font-weight:800;"
                     "letter-spacing:.04em;padding:2px 9px;border-radius:999px;margin-left:10px'>FOCUSED</span>"
                     if focus == cname else "")
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:.2rem'>"
                f"<span style='width:13px;height:13px;border-radius:50%;background:{color};"
                f"display:inline-block;flex:none'></span>"
                f"<span style='font-size:1.18rem;font-weight:800;color:#2B2F36'>{cname}</span>"
                f"{badge}</div>", unsafe_allow_html=True)

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Customers",          f"{len(seg):,}")
            m2.metric("Share of base",      f"{share:.0f}%")
            m3.metric("Avg lifetime spend", f"€{seg['total_spend'].mean():,.0f}")
            m4.metric("Segment revenue",    f"€{revenue/1e6:,.1f}M",
                      delta=f"{rev_pct:.0f}% of total", delta_color="off")
            m5.metric("Promo purchases",    f"{promo:.0f}%")

            desc = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", describe_segment(df_info, cid))
            st.markdown(f"<p style='color:#6B7079;margin:.4rem 0 .2rem'>{desc}</p>",
                        unsafe_allow_html=True)

            left, right = st.columns([1.25, 1])
            with left:
                st.markdown("**Recommended campaigns**")
                for campaign in suggest_campaigns(df_info, cid):
                    html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", campaign)
                    st.markdown(
                        f"<div style='border:1px solid #ECECEF;border-left:3px solid {color};"
                        f"border-radius:10px;padding:.55rem .85rem;margin:.35rem 0;background:#fff;"
                        f"color:#3B3F46;font-size:.92rem;line-height:1.4'>{html}</div>",
                        unsafe_allow_html=True)
            with right:
                seg_spend = seg[SPEND_COLS].mean().sort_values(ascending=True)
                fig = px.bar(
                    x=seg_spend.values,
                    y=[c.replace("lifetime_spend_", "").replace("_", " ").title() for c in seg_spend.index],
                    orientation="h", title="Avg spend per category (€)",
                    labels={"x": "", "y": ""}, color_discrete_sequence=[color])
                fig.update_traces(hovertemplate="<b>%{y}</b><br>€%{x:,.0f} avg<extra></extra>")
                show(fig, key=f"camp_{cid}", height=300, lock=True, value_axis="x")

            if run_rules:
                rules_df = get_rules_for_cluster(df_info, df_basket, cid)
                if not rules_df.empty:
                    st.markdown("**Top product associations**")
                    disp = rules_df[["antecedents", "consequents", "support", "confidence", "lift"]].copy()
                    disp["antecedents"] = disp["antecedents"].apply(lambda x: ", ".join(list(x)))
                    disp["consequents"] = disp["consequents"].apply(lambda x: ", ".join(list(x)))
                    st.dataframe(disp, width="stretch")
        st.write("")
