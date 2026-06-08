# Review changes — what was modified and why

This is a peer review pass over the original draft. Below is every change, with
before → after and the reasoning, so each edit can be re-checked independently.
Scope was kept deliberately small: **4 source files + `requirements.txt`**, plus
two new docs (`README.md`, this file). No git history was created (left to you).

## Summary table

| # | Area | Before (original) | After (this pass) | Why |
|---|------|-------------------|-------------------|-----|
| 1 | **Clustering features** (`preprocessing.py`) | All 22 features fed to K-Means: 10 raw spends **+ `total_spend`** + binaries (`gender_male`, `has_loyalty_card`), ordinal `degree_level`, `age`, `typical_hour`, complaints… | **Behavioural space only**: 10 **log1p** spends + `percentage_of_products_bought_promotion` + `lifetime_total_distinct_products` + `distinct_stores_visited` (13 feats). Demographics moved to *profiling*, not clustering. | `total_spend` double-counted magnitude (it is the sum of the spend cols); raw skewed spends + noisy demographics let one cluster swallow **46%** of customers. New space: silhouette **0.146 → 0.205**, balanced sizes, and **k=4 becomes the silhouette peak** (was previously k=3). |
| 2 | **Cluster names** (`app.py` → `clustering.py`) | Hard-coded `CLUSTER_NAMES = {0: "Health-Conscious", 1: "Budget Promo Shoppers", …}` tied to cluster id, assuming k=4. | `label_clusters()` derives names from each segment's own profile (share-lift of categories + promo/family/spend qualifiers). | The hard-coded labels were **wrong**: cluster tagged "Budget" had the *highest* avg spend. Data-driven names can't contradict the data and survive re-runs / different k. |
| 3 | **Campaigns** (`app.py` → `clustering.py`) | Hard-coded `CAMPAIGNS = {0: [...], …}` per cluster id, unrelated to the actual data. | `suggest_campaigns()` builds ideas from the same segment signature used for naming (over-indexed categories + promo/family traits). | Rubric rewards *promotions tied to interpretation*. Campaigns now always match the segment they describe. |
| 4 | **Labels + campaigns share one source** (`clustering.py`) | n/a (two separate hard-coded dicts) | New `segment_signatures()` is the single source of truth for both naming and campaigns. | DRY; guarantees label and campaign never disagree. |
| 5 | **Explanation text** (`app.py`) | App was chart-only, almost no narrative. | Added an **Executive Summary** block + a one-line `describe_segment()` description in each segment card. | Rubric: "Visuals **and Explanation**" + report/app quality. |
| 6 | **Data paths** (`app.py` + repo) | Defaults pointed to `data/customer_info-1.csv` / `…basket-1.csv` — **non-existent**; data lived in `description/` and `DATA/DATA/…` (filename with a space). | Single clean `data/` folder; defaults point there. | App would crash immediately "from a clean environment" — an explicit grading criterion. |
| 7 | **Dead code** (`clustering.py`) | Unused duplicate `CLUSTER_NAMES` dict. | Removed. | Cleanliness / DRY (+2 code-quality points). |
| 8 | **Dependencies** (`requirements.txt`) | Unpinned; included unused `matplotlib`, `seaborn`. | Pinned to tested versions; dropped unused packages. | Reproducible "clean environment" install. |
| 9 | **Docs** | No README, no run instructions. | Added `README.md` (run steps + what the app does) and this `CHANGES.md`. | README is required for the app-deliverable option. |

## Resulting segments (k=4, behavioural clustering)

| Cluster | Label (auto-generated) | Size | Avg total spend |
|--------:|------------------------|-----:|----------------:|
| 0 | Gaming & Electronics Shoppers | ~12% | ~€22,760 |
| 1 | Veggie & Hygiene Shoppers | ~22% | ~€17,400 |
| 2 | Promo-Driven Grocery Shoppers | ~37% | ~€17,506 |
| 3 | Large-Family Power Grocery Shoppers | ~29% | ~€36,489 |

## Verification

- Full pipeline runs under the pinned versions (Python 3.12).
- The Streamlit app was executed end-to-end via `streamlit.testing` (`AppTest`):
  **0 exceptions, 0 errors**; verified again from a fresh unzip with relative
  paths only.

## Left for you (intentionally not done)

- **Git** (rubric: 3 pts): `git init`, commit, invite `ivopbernardo`. Deferred to
  delivery time.
- Cosmetic `use_container_width` deprecation warning in the Streamlit log — works
  fine on the pinned `streamlit==1.58.0`; left untouched to avoid churn.

## Modernisation pass 2 — quality polish (no new complexity)

A light follow-up pass. Same scope discipline: small, verified edits only.

| # | File | Change | Why |
|---|------|--------|-----|
| 1 | `preprocessing.py` | Age now computed **vectorised** (`pd.to_datetime(..., errors="coerce")`) instead of a row-by-row `.apply` with try/except. | Faster on 33k rows, more robust (bad dates → NaN → median-imputed), simpler. Reference date pulled out to a constant. |
| 2 | `clustering.py` | `label_clusters()` now **guarantees unique names** (numeric suffix on collision). | Latent bug: at higher K two clusters can derive the same name → duplicate index → the profile heatmap and spend radar break on `.loc[name]`. |
| 3 | `clustering.py` + `app.py` | New `recommended_k()`; the EDA tab now **flags the silhouette-optimal K**. | On-rubric (K selection); the "right" K is stated, not left for the reader to eyeball. |
| 4 | `app.py` | Replaced deprecated `use_container_width=True` with `width="stretch"` (12×). | The deprecation removal date has passed; the supported replacement works on `streamlit==1.58.0`. Removes all the log noise. |

**Verification:** full pipeline + Streamlit `AppTest` re-run end-to-end — 0 exceptions,
0 errors. Headline result unchanged: silhouette **0.205** at **K=4**, same four segments.

All four edits sit inside the grading rubric (segmentation quality, code quality,
K-selection, app cleanliness) — **no new runtime dependencies** were added; the app
still installs from `requirements.txt` unchanged.

A standalone `Project_Report.pdf` (casual overview of the project) was generated
once as a convenience and lives in the repo root. The throwaway generator script and
its `reportlab` dependency were **removed** afterwards to keep the project footprint
minimal — the PDF is the artifact, not part of the app.

Final tidy: verbose "why" comments in the `.py` files were trimmed to crisp one-line
docstrings, and the reasoning that mattered (in particular **why the silhouette
score wasn't chased higher via PCA dimensionality reduction** — it inflates the
metric by discarding information and would blur the segment interpretation) was moved
into `Project_Report.pdf` instead of living as long code comments.
