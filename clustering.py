import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from preprocessing import SPEND_COLS


# Friendly, short label for each spend category — used to name segments.
CATEGORY_LABELS = {
    "lifetime_spend_groceries":         "Grocery",
    "lifetime_spend_electronics":       "Electronics",
    "lifetime_spend_vegetables":        "Veggie",
    "lifetime_spend_nonalcohol_drinks": "Soft-Drink",
    "lifetime_spend_alcohol_drinks":    "Alcohol",
    "lifetime_spend_meat":              "Meat",
    "lifetime_spend_fish":              "Fish",
    "lifetime_spend_hygiene":           "Hygiene",
    "lifetime_spend_videogames":        "Gaming",
    "lifetime_spend_petfood":           "Pet",
}


def evaluate_k(X_scaled, k_range=range(2, 9), random_state=42) -> pd.DataFrame:
    """Return DataFrame with inertia and silhouette score for each k."""
    rows = []
    for k in k_range:
        km     = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        rows.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(
                X_scaled, labels, sample_size=5000, random_state=random_state
            ),
        })
    return pd.DataFrame(rows).set_index("k")


def recommended_k(k_eval: pd.DataFrame) -> int:
    """The K with the highest silhouette score — the data's own preference."""
    return int(k_eval["silhouette"].idxmax())


def apply_kmeans(X_scaled, n_clusters: int = 4, random_state: int = 42):
    """Fit KMeans. Returns (cluster_labels, pca_df, kmeans_model)."""
    km     = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=15)
    labels = km.fit_predict(X_scaled)

    pca        = PCA(n_components=2, random_state=random_state)
    components = pca.fit_transform(X_scaled)
    pca_df     = pd.DataFrame(components, columns=["PCA1", "PCA2"])
    pca_df["Cluster"] = labels

    return labels, pca_df, km


def generate_cluster_profiles(df: pd.DataFrame, cluster_col: str = "Cluster") -> pd.DataFrame:
    """Return mean of numeric features per cluster."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if cluster_col in numeric_cols:
        numeric_cols.remove(cluster_col)
    return df.groupby(cluster_col)[numeric_cols].mean().round(2)


def segment_signatures(df: pd.DataFrame, cluster_col: str = "Cluster") -> dict:
    """Single source of truth for each segment: core categories (share-lift vs.
    the overall mix) plus promo / family / spend qualifiers. Both names and
    campaigns derive from this, so they always agree and nothing is hard-coded."""
    prof    = df.groupby(cluster_col)[SPEND_COLS].mean()
    shares  = prof.div(prof.sum(axis=1), axis=0)
    overall = prof.sum() / prof.sum().sum()
    lift    = shares.div(overall, axis=1)

    def _z(col):
        s = df.groupby(cluster_col)[col].mean()
        return (s - s.mean()) / (s.std() or 1.0), s

    promo_z, promo  = _z("percentage_of_products_bought_promotion")
    kids_z, kids    = _z("total_kids")
    total           = df.groupby(cluster_col)["total_spend"].mean()

    sigs = {}
    for c in prof.index:
        cats = [(k, lift.loc[c, k]) for k in SPEND_COLS
                if shares.loc[c, k] > 0.05 and lift.loc[c, k] > 1.3]
        cats.sort(key=lambda x: -x[1])
        core_cols = [k for k, _ in cats[:2]] or [shares.loc[c].idxmax()]
        sigs[c] = {
            "core":         [CATEGORY_LABELS[k] for k in core_cols],
            "promo_high":   bool(promo_z[c] > 0.7),
            "promo_rate":   float(promo[c]),
            "family_large": bool(kids_z[c] > 0.7),
            "kids":         float(kids[c]),
            "top_spender":  bool(total[c] == total.max()),
        }
    return sigs


def label_clusters(df: pd.DataFrame, cluster_col: str = "Cluster") -> dict:
    """Derive a human-readable name for each cluster from its profile signature."""
    names, seen = {}, {}
    for c, s in segment_signatures(df, cluster_col).items():
        qualifiers = []
        if s["promo_high"]:
            qualifiers.append("Promo-Driven")
        if s["family_large"]:
            qualifiers.append("Large-Family")
        if s["top_spender"]:
            qualifiers.append("Power")
        name = (" ".join(qualifiers) + " " + " & ".join(s["core"]) + " Shoppers").strip()
        # Keep names unique (two clusters can derive the same one at higher K).
        if name in seen:
            seen[name] += 1
            name = f"{name} ({seen[name]})"
        else:
            seen[name] = 1
        names[c] = name
    return names


def describe_segment(df: pd.DataFrame, cluster_id: int, cluster_col: str = "Cluster") -> str:
    """One-line plain-English description of a segment, built from its signature."""
    s     = segment_signatures(df, cluster_col)[cluster_id]
    seg   = df[df[cluster_col] == cluster_id]
    share = len(seg) / len(df) * 100
    spend = seg["total_spend"].mean()

    parts = [f"**{share:.0f}%** of customers", f"~€{spend:,.0f} avg lifetime spend"]
    parts.append("over-indexes on " + " & ".join(s["core"]).lower())
    if s["promo_high"]:
        parts.append(f"highly promotion-driven ({s['promo_rate']*100:.0f}% of buys on promo)")
    if s["family_large"]:
        parts.append(f"large families (~{s['kids']:.1f} kids/teens at home)")
    if s["top_spender"]:
        parts.append("the highest-value segment")
    return "; ".join(parts) + "."


def suggest_campaigns(df: pd.DataFrame, cluster_id: int, cluster_col: str = "Cluster",
                      top_n: int = 3) -> list:
    """Campaign ideas from the segment's signature, so promotions always match
    its label: defining traits (family, promo) lead, then cross-sell / loyalty."""
    s    = segment_signatures(df, cluster_col)[cluster_id]
    core = s["core"]
    ideas = []

    if s["family_large"]:
        ideas.append("📦 **Family Bulk Deal** — 15% off when the basket exceeds €120")
    if len(core) >= 2:
        ideas.append(f"🔗 **{core[0]} + {core[1]} Bundle** — buy both together, "
                     f"get 20% off the cheaper one")
    ideas.append(f"⭐ **{core[0]} Loyalty Boost** — double loyalty points on all "
                 f"{core[0].lower()} purchases this month")
    if s["promo_high"]:
        ideas.append("🏷️ **Weekly Promo Drop** — exclusive extra discount on top categories "
                     "every week (this segment buys heavily on promotion)")
    return ideas[:top_n]
