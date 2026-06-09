# Customer Segmentation & Targeted Campaigns — ML II (NOVA IMS)

Interactive Streamlit app that segments customers from `customer_info` /
`customer_basket`, profiles each segment, and proposes data-driven marketing
campaigns. Delivered as a runnable app (the alternative to a PDF report).

## Run

Requires **Python 3.12**. From the project root:

```bash
# 1. (recommended) create and activate an isolated environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. install the pinned dependencies
pip install -r requirements.txt

# 3. launch
streamlit run app.py
```

Streamlit opens the app in your browser automatically (usually at
<http://localhost:8501>; if it doesn't open, paste that URL manually).

**The datasets are included** — `data/customer_info.csv` and
`data/customer_basket.csv` ship with the repo, so the app runs straight after
cloning, no extra files needed. The app loads them automatically; you can also
point it at other CSVs from the sidebar.

## What it does

- **EDA** — distributions of age, spend, gender, education, and K selection
  (elbow + silhouette).
- **Segmentation** — K-Means on a **behavioural** feature space: log-scaled
  per-category spend + promotion rate + product/store breadth. Demographics are
  deliberately left out of the distance metric and used afterwards to profile
  the segments. Silhouette peaks at **K = 4**.
- **Cluster profiles** — mean spend per category, key demographics, spend radar.
- **Campaigns** — each segment is named and described **from its own profile**
  (no hard-coded labels), and campaign ideas are generated from the segment's
  over-indexed categories and promo/family traits. Optional per-segment
  association rules (mlxtend / Apriori) can be mined from the basket data.

The sidebar also exports `customer_clusters.csv` (every `customer_id` with its
assigned cluster).

## Code layout

| File | Responsibility |
|------|----------------|
| `preprocessing.py` | load + clean `customer_info`, feature engineering, behavioural matrix |
| `clustering.py`    | K selection, K-Means, profiles, data-driven labels & campaigns |
| `market_basket.py` | basket loading + association rules |
| `app.py`           | Streamlit UI |
