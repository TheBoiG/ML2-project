import ast
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules as ar


def load_baskets(basket_path):
    df = pd.read_csv(basket_path)
    df["list_of_goods"] = df["list_of_goods"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    return df


def get_association_rules(baskets, min_support=0.01, min_lift=1.2):
    transactions = baskets["list_of_goods"].tolist()
    te = TransactionEncoder()
    te_array = te.fit_transform(transactions)
    df_enc = pd.DataFrame(te_array, columns=te.columns_)
    frequent = apriori(df_enc, min_support=min_support, use_colnames=True)
    rules = ar(frequent, metric="lift", min_threshold=min_lift)
    return rules.sort_values(["lift", "confidence"], ascending=False)


def get_rules_for_cluster(df_info, df_basket, cluster_id, cluster_col="Cluster",
                           min_support=0.008, min_lift=1.2, top_n=5):
    cust_ids = set(df_info.loc[df_info[cluster_col] == cluster_id, "customer_id"])
    sub = df_basket[df_basket["customer_id"].isin(cust_ids)]
    if len(sub) < 100:
        return pd.DataFrame()
    try:
        rules = get_association_rules(sub, min_support=min_support, min_lift=min_lift)
        return rules.head(top_n)
    except Exception:
        return pd.DataFrame()
