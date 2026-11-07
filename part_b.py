"""Build Part B customer segments and a leakage-controlled lapse model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
CUTOFF = pd.Timestamp("2025-09-01 00:00:00")
WINDOW_END = pd.Timestamp("2025-10-01 00:00:00")
NUMERIC_FEATURES = [
    "recency",
    "frequency",
    "monetary",
    "average_discount_pct",
    "cancellation_rate",
    "average_delivery_minutes",
    "age",
    "tenure_days",
]
CATEGORICAL_FEATURES = ["city", "gender", "favourite_category"]


def load_clean_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    customers = pd.read_csv(DATA / "clean" / "customers.csv", parse_dates=["signup_date"])
    products = pd.read_csv(DATA / "clean" / "products.csv")
    orders = pd.read_csv(DATA / "clean" / "orders.csv", parse_dates=["order_ts"])
    return customers, products, orders


def rfm_table(orders: pd.DataFrame) -> pd.DataFrame:
    completed = orders.loc[orders["status"].eq("completed")].copy()
    rfm = completed.groupby("customer_id").agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
    )
    rfm["recency"] = (SNAPSHOT - rfm["last_order_ts"]).dt.days
    return rfm.drop(columns="last_order_ts")[
        ["recency", "frequency", "monetary"]
    ].reset_index()


def segment_customers(rfm: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rfm_columns = ["recency", "frequency", "monetary"]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm[rfm_columns])

    scores = []
    for k in range(2, 9):
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = model.fit_predict(scaled)
        scores.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                "silhouette": float(silhouette_score(scaled, labels)),
            }
        )

    chosen_k = 4
    final_model = KMeans(n_clusters=chosen_k, n_init=10, random_state=42)
    segmented = rfm.copy()
    segmented["segment_id"] = final_model.fit_predict(scaled)
    profiles = segmented.groupby("segment_id").agg(
        size=("customer_id", "count"),
        mean_recency=("recency", "mean"),
        mean_frequency=("frequency", "mean"),
        mean_monetary=("monetary", "mean"),
    )

    premium_id = int(profiles["mean_monetary"].idxmax())
    remaining = [int(value) for value in profiles.index if value != premium_id]
    frequent_id = int(profiles.loc[remaining, "mean_frequency"].idxmax())
    remaining.remove(frequent_id)
    at_risk_id = int(profiles.loc[remaining, "mean_recency"].idxmax())
    core_id = next(value for value in remaining if value != at_risk_id)
    names = {
        premium_id: "Premium High-Value",
        frequent_id: "Frequent Essentials",
        at_risk_id: "Cooling-Off Occasional",
        core_id: "Steady Everyday",
    }
    actions = {
        "Premium High-Value": "Offer a tailored premium bundle and protect the relationship with proactive service.",
        "Frequent Essentials": "Use replenishment reminders and practical multi-buy offers.",
        "Cooling-Off Occasional": "Send a modest win-back offer and test whether delivery experience is a barrier.",
        "Steady Everyday": "Encourage a second category with a relevant cross-category recommendation.",
    }
    segmented["segment_name"] = segmented["segment_id"].map(names)
    segmented["is_premium"] = segmented["segment_id"].eq(premium_id)
    profile_rows = []
    for segment_id, row in profiles.sort_index().iterrows():
        name = names[int(segment_id)]
        profile_rows.append(
            {
                "segment_id": int(segment_id),
                "segment_name": name,
                "size": int(row["size"]),
                "mean_recency": float(row["mean_recency"]),
                "mean_frequency": float(row["mean_frequency"]),
                "mean_monetary": float(row["mean_monetary"]),
                "is_premium": bool(int(segment_id) == premium_id),
                "recommended_action": actions[name],
            }
        )

    pca = PCA(n_components=2)
    pca.fit(scaled)
    segmented = segmented[
        [
            "customer_id",
            "recency",
            "frequency",
            "monetary",
            "segment_id",
            "segment_name",
            "is_premium",
        ]
    ]
    segmented["is_premium"] = segmented["is_premium"].astype(bool)
    return segmented, {
        "chosen_k": chosen_k,
        "kmeans_scores": scores,
        "profiles": profile_rows,
        "premium_segment_id": premium_id,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "pca_total_explained_variance_ratio": float(pca.explained_variance_ratio_.sum()),
    }


def lapse_dataset(
    customers: pd.DataFrame, products: pd.DataFrame, orders: pd.DataFrame
) -> pd.DataFrame:
    completed_before = orders.loc[
        orders["status"].eq("completed") & orders["order_ts"].lt(CUTOFF)
    ].copy()
    final_before = orders.loc[
        orders["status"].isin(["completed", "cancelled"]) & orders["order_ts"].lt(CUTOFF)
    ].copy()
    september_completed = orders.loc[
        orders["status"].eq("completed")
        & orders["order_ts"].ge(CUTOFF)
        & orders["order_ts"].lt(WINDOW_END)
    ]

    # The cohort is defined only by completed orders strictly before the cutoff.
    cohort_ids = sorted(completed_before["customer_id"].unique())
    completed_before = completed_before.merge(
        products[["product_id", "category"]],
        on="product_id",
        how="left",
        validate="many_to_one",
    )
    behaviour = completed_before.groupby("customer_id").agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
        average_discount_pct=("discount_pct", "mean"),
        average_delivery_minutes=("delivery_minutes", "mean"),
    )
    behaviour["recency"] = (CUTOFF - behaviour["last_order_ts"]).dt.days
    behaviour = behaviour.drop(columns="last_order_ts")

    # Resolve tied favourite categories alphabetically for a stable, documented result.
    category_counts = (
        completed_before.groupby(["customer_id", "category"], as_index=False)
        .size()
        .sort_values(["customer_id", "size", "category"], ascending=[True, False, True])
        .drop_duplicates("customer_id", keep="first")
        .set_index("customer_id")["category"]
        .rename("favourite_category")
    )
    final_counts = final_before.groupby("customer_id").size().rename("final_order_count")
    cancelled_counts = (
        final_before.loc[final_before["status"].eq("cancelled")]
        .groupby("customer_id")
        .size()
        .rename("cancelled_order_count")
    )
    cancellation = pd.concat([final_counts, cancelled_counts], axis=1).fillna(0)
    cancellation["cancellation_rate"] = (
        cancellation["cancelled_order_count"] / cancellation["final_order_count"]
    )

    september_ids = set(september_completed["customer_id"])
    data = behaviour.join(category_counts).join(cancellation[["cancellation_rate"]])
    data = data.loc[cohort_ids].copy()
    data["lapsed"] = [int(customer_id not in september_ids) for customer_id in data.index]
    data.index.name = "customer_id"
    data = data.reset_index().merge(
        customers[["customer_id", "city", "age", "gender", "signup_date"]],
        on="customer_id",
        how="left",
        validate="one_to_one",
    )
    data["tenure_days"] = (CUTOFF - data["signup_date"]).dt.days
    return data[
        [
            "customer_id",
            "recency",
            "frequency",
            "monetary",
            "average_discount_pct",
            "cancellation_rate",
            "average_delivery_minutes",
            "favourite_category",
            "city",
            "age",
            "gender",
            "tenure_days",
            "lapsed",
        ]
    ]


def metric_summary(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix_labels_0_1": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "precision_lapsed": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_lapsed": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_lapsed": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }


def make_pipeline(model) -> Pipeline:
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return Pipeline([("preprocessor", preprocessing), ("model", model)])


def main() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    customers, products, orders = load_clean_data()
    orders["order_ts"] = pd.to_datetime(orders["order_ts"])

    rfm = rfm_table(orders)
    segments, segment_report = segment_customers(rfm)
    segments.to_csv(OUTPUTS / "customer_segments.csv", index=False)

    cohort = lapse_dataset(customers, products, orders)
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = cohort[feature_columns].copy()
    y = cohort["lapsed"].astype(int)
    assert not X.isna().any().any(), "A model feature contains a missing value."
    assert not set(X.columns) & {"segment_id", "segment_name", "is_premium"}

    indices = np.arange(len(cohort))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    assert set(train_idx).isdisjoint(set(test_idx))

    baseline_predictions = np.zeros(len(y_test), dtype=int)
    baseline = metric_summary(y_test, baseline_predictions)

    logistic_search = GridSearchCV(
        make_pipeline(LogisticRegression(max_iter=1000)),
        {"model__C": [0.01, 0.1, 1, 10, 100]},
        cv=5,
        scoring="f1",
        n_jobs=1,
        refit=True,
    )
    logistic_search.fit(X_train, y_train)

    forest_search = GridSearchCV(
        make_pipeline(RandomForestClassifier(n_estimators=100, random_state=42)),
        {
            "model__max_depth": [3, 5, 8],
            "model__min_samples_leaf": [5, 10, 20],
        },
        cv=5,
        scoring="f1",
        n_jobs=1,
        refit=True,
    )
    forest_search.fit(X_train, y_train)

    logistic_test = metric_summary(y_test, logistic_search.best_estimator_.predict(X_test))
    forest_test = metric_summary(y_test, forest_search.best_estimator_.predict(X_test))
    # Select by cross-validated F1; use the simpler logistic model for a tie.
    selected_name = (
        "logistic_regression"
        if logistic_search.best_score_ >= forest_search.best_score_
        else "random_forest"
    )
    selected_model = (
        logistic_search.best_estimator_
        if selected_name == "logistic_regression"
        else forest_search.best_estimator_
    )

    # Compare the training-only scaler mean against the full cohort's descriptive mean.
    fitted_scaler = selected_model.named_steps["preprocessor"].named_transformers_["numeric"]
    frequency_position = NUMERIC_FEATURES.index("frequency")
    train_scaler_frequency_mean = float(fitted_scaler.mean_[frequency_position])
    full_cohort_frequency_mean = float(cohort["frequency"].mean())

    # Oversample only the original training rows; the unchanged test rows are never sampled.
    majority_count = int(y_train.value_counts().max())
    minority_indices = y_train.index[y_train.eq(1)].to_numpy()
    additions_needed = majority_count - len(minority_indices)
    random_state = np.random.RandomState(42)
    sampled_indices = random_state.choice(minority_indices, size=additions_needed, replace=True)
    X_balanced = pd.concat([X_train, X_train.loc[sampled_indices]], ignore_index=True)
    y_balanced = pd.concat(
        [y_train.reset_index(drop=True), y_train.loc[sampled_indices].reset_index(drop=True)],
        ignore_index=True,
    )
    # Refit a fresh copy so preprocessing also sees only the balanced training rows.
    from sklearn.base import clone

    balanced_logistic = clone(logistic_search.best_estimator_)
    balanced_logistic.fit(X_balanced, y_balanced)
    balanced_lr_test = metric_summary(y_test, balanced_logistic.predict(X_test))

    # Feature interpretation comes from the best tuned models, not the chosen test outcome.
    best_forest = forest_search.best_estimator_
    forest_names = best_forest.named_steps["preprocessor"].get_feature_names_out()
    forest_values = best_forest.named_steps["model"].feature_importances_
    top_importance_indices = np.argsort(forest_values)[::-1][:5]
    top_forest_features = [
        {"feature": str(forest_names[index]), "importance": float(forest_values[index])}
        for index in top_importance_indices
    ]

    best_logistic = logistic_search.best_estimator_
    logistic_names = best_logistic.named_steps["preprocessor"].get_feature_names_out()
    logistic_coefficients = best_logistic.named_steps["model"].coef_[0]
    coefficient_interpretations = []
    for feature in ["numeric__recency", "numeric__frequency", "numeric__monetary"]:
        position = list(logistic_names).index(feature)
        coefficient = float(logistic_coefficients[position])
        direction = "raises" if coefficient > 0 else "lowers" if coefficient < 0 else "does not change"
        coefficient_interpretations.append(
            {"feature": feature.removeprefix("numeric__"), "coefficient": coefficient, "direction": direction}
        )

    model_path = OUTPUTS / "lapse_model.joblib"
    joblib.dump(selected_model, model_path)
    reloaded_model = joblib.load(model_path)
    original_test_predictions = selected_model.predict(X_test)
    reloaded_test_predictions = reloaded_model.predict(X_test)
    predictions_reproduced = bool(np.array_equal(original_test_predictions, reloaded_test_predictions))
    assert predictions_reproduced

    # The test-set IDs are aligned with the untouched feature rows from train_test_split.
    test_customers = cohort.iloc[test_idx][["customer_id", "lapsed"]].reset_index(drop=True)
    premium_lookup = segments.set_index("customer_id")["is_premium"]
    test_customers["is_premium"] = test_customers["customer_id"].map(premium_lookup)
    assert test_customers["is_premium"].notna().all()
    premium_mask = test_customers["is_premium"].to_numpy(dtype=bool)
    test_probabilities = selected_model.predict_proba(X_test)[:, list(selected_model.classes_).index(1)]
    retention = pd.DataFrame(
        {
            "customer_id": test_customers.loc[premium_mask, "customer_id"].to_numpy(),
            "lapse_probability": test_probabilities[premium_mask],
            "actually_lapsed": test_customers.loc[premium_mask, "lapsed"].to_numpy(dtype=int),
        }
    ).sort_values(["lapse_probability", "customer_id"], ascending=[False, True]).head(10)
    retention["lapse_probability"] = retention["lapse_probability"].round(6)
    retention.to_csv(OUTPUTS / "retention_list.csv", index=False)

    rfm_stats = {
        "customers": int(len(rfm)),
        "recency_median": float(rfm["recency"].median()),
        "recency_min": int(rfm["recency"].min()),
        "recency_max": int(rfm["recency"].max()),
        "frequency_mean": float(rfm["frequency"].mean()),
        "frequency_max": int(rfm["frequency"].max()),
        "monetary_mean": float(rfm["monetary"].mean()),
        "monetary_max": float(rfm["monetary"].max()),
    }
    report = {
        "rfm": rfm_stats,
        "segmentation": segment_report,
        "lapse_cohort": {
            "customers": int(len(cohort)),
            "lapsed": int(y.sum()),
            "lapsed_percent": float(y.mean() * 100),
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "features": feature_columns,
            "max_order_date_entering_features": "2025-08-31",
            "segment_features_excluded": True,
            "features_have_missing_values": bool(X.isna().any().any()),
            "favourite_category_tie_rule": "alphabetically first category",
        },
        "baseline": baseline,
        "logistic_regression": {
            "best_params": logistic_search.best_params_,
            "best_cv_f1": float(logistic_search.best_score_),
            "test": logistic_test,
        },
        "random_forest": {
            "best_params": forest_search.best_params_,
            "best_cv_f1": float(forest_search.best_score_),
            "test": forest_test,
        },
        "selected_model": selected_name,
        "frequency_scaler_mean_train_only": train_scaler_frequency_mean,
        "frequency_mean_full_cohort": full_cohort_frequency_mean,
        "class_balance": {
            "train_counts_before": {str(k): int(v) for k, v in y_train.value_counts().sort_index().items()},
            "train_counts_after": {str(k): int(v) for k, v in y_balanced.value_counts().sort_index().items()},
            "added_lapsed_duplicates": int(additions_needed),
            "balanced_logistic_test": balanced_lr_test,
        },
        "random_forest_top_features": top_forest_features,
        "logistic_coefficient_signs": coefficient_interpretations,
        "saved_model_predictions_reproduced": predictions_reproduced,
        "retention_list": {
            "rows": int(len(retention)),
            "actually_lapsed_count": int(retention["actually_lapsed"].sum()),
            "premium_test_customers": int(premium_mask.sum()),
            "output_columns": list(retention.columns),
        },
    }
    (OUTPUTS / "part_b_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
