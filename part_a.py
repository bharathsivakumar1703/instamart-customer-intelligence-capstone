"""Run the Part A raw-data audit, cleaning, storage, and analysis."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
DB_PATH = ROOT / "data" / "instamart.db"
SQL_PATH = ROOT / "sql" / "queries.sql"


def audit_table(df: pd.DataFrame) -> dict:
    missing = df.isna().sum()
    return {
        "shape": list(df.shape),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "missing": {
            column: {
                "count": int(missing[column]),
                "percent": round(float(missing[column] / len(df) * 100), 2),
            }
            for column in df.columns
        },
    }


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    customers = pd.read_csv(RAW / "customers.csv")
    products = pd.read_csv(RAW / "products.csv")
    orders = pd.read_csv(RAW / "orders.csv")
    return customers, products, orders


def clean_customers(raw: pd.DataFrame) -> pd.DataFrame:
    customers = raw.copy()
    customers["city"] = customers["city"].astype("string").str.strip().str.title()

    gender = customers["gender"].fillna("").astype("string").str.strip().str.casefold()
    customers["gender"] = gender.map(
        {"male": "male", "m": "male", "female": "female", "f": "female"}
    ).fillna("unknown")

    ages = pd.to_numeric(customers["age"], errors="coerce")
    ages = ages.where(ages.between(10, 100))
    city_medians = ages.groupby(customers["city"]).transform("median")
    customers["age"] = ages.fillna(city_medians)
    customers["signup_date"] = pd.to_datetime(customers["signup_date"], errors="raise")
    return customers


def clean_orders(raw: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    orders = raw.copy()
    orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="raise")
    orders = orders.drop_duplicates(keep="first")
    orders = orders.sort_values(["order_ts"], kind="stable").drop_duplicates(
        subset=["order_id"], keep="last"
    )

    orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")
    orders["delivery_minutes"] = pd.to_numeric(
        orders["delivery_minutes"], errors="coerce"
    )
    orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = np.nan

    product_lookup = products.set_index("product_id")
    orders["category"] = orders["product_id"].map(product_lookup["category"])
    orders["price_inr"] = orders["product_id"].map(product_lookup["price_inr"])
    missing_amount = orders["order_amount"].isna()
    orders.loc[missing_amount, "order_amount"] = (
        orders.loc[missing_amount, "price_inr"]
        * orders.loc[missing_amount, "quantity"]
        * (1 - orders.loc[missing_amount, "discount_pct"] / 100)
    ).round(2)

    completed = orders["status"].eq("completed")
    completed_medians = orders.loc[completed].groupby("category")[
        "delivery_minutes"
    ].transform("median")
    orders.loc[completed, "delivery_minutes"] = orders.loc[
        completed, "delivery_minutes"
    ].fillna(completed_medians)

    orders = orders.drop(columns=["category", "price_inr"])
    orders["order_ts"] = orders["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return orders.sort_values("order_id", kind="stable").reset_index(drop=True)


def raw_audit(
    customers: pd.DataFrame, products: pd.DataFrame, orders: pd.DataFrame
) -> dict:
    raw_age = pd.to_numeric(customers["age"], errors="coerce")
    raw_city = customers["city"].astype("string")
    raw_gender = customers["gender"].astype("string")
    numeric_amounts = pd.to_numeric(orders["order_amount"], errors="coerce")
    numeric_delivery = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
    return {
        "tables": {
            "customers": audit_table(customers),
            "products": audit_table(products),
            "orders": audit_table(orders),
        },
        "figures": {
            "orders_rows": int(len(orders)),
            "exact_duplicate_rows": int(orders.duplicated().sum()),
            "pending_rows": int(orders["status"].eq("pending").sum()),
            "distinct_order_ids": int(orders["order_id"].nunique()),
            "non_numeric_amounts": int(numeric_amounts.isna().sum()),
            "negative_delivery_minutes": int(numeric_delivery.lt(0).sum()),
            "blank_delivery_minutes": int(orders["delivery_minutes"].isna().sum()),
            "blank_ages": int(customers["age"].isna().sum()),
            "ages_outside_10_100": int((raw_age.notna() & ~raw_age.between(10, 100)).sum()),
            "distinct_raw_city_spellings": int(raw_city.nunique(dropna=True)),
            "distinct_nonblank_raw_gender_spellings": int(
                raw_gender[raw_gender.notna() & raw_gender.str.strip().ne("")].nunique()
            ),
            "blank_genders": int(customers["gender"].isna().sum()),
        },
    }


def cleaning_changes(
    customers_raw: pd.DataFrame,
    customers_clean: pd.DataFrame,
    products_raw: pd.DataFrame,
    orders_raw: pd.DataFrame,
    orders_clean: pd.DataFrame,
) -> dict:
    normalized_city = customers_raw["city"].astype("string").str.strip().str.title()
    raw_gender = customers_raw["gender"].fillna("").astype("string")
    numeric_age = pd.to_numeric(customers_raw["age"], errors="coerce")
    invalid_age = numeric_age.notna() & ~numeric_age.between(10, 100)

    orders_stage = orders_raw.copy()
    orders_stage["order_ts"] = pd.to_datetime(orders_stage["order_ts"], errors="raise")
    exact_duplicate_rows = int(orders_stage.duplicated().sum())
    after_exact_dedup = orders_stage.drop_duplicates(keep="first")
    after_latest = after_exact_dedup.sort_values(["order_ts"], kind="stable").drop_duplicates(
        subset=["order_id"], keep="last"
    )
    numeric_amount = pd.to_numeric(after_latest["order_amount"], errors="coerce")
    numeric_delivery = pd.to_numeric(after_latest["delivery_minutes"], errors="coerce")
    negative_delivery = numeric_delivery.lt(0)
    missing_completed_delivery = after_latest["status"].eq("completed") & (
        numeric_delivery.isna() | negative_delivery
    )

    return {
        "customers": {
            "city_strings_standardized": int(customers_raw["city"].ne(normalized_city).sum()),
            "gender_values_mapped": int(raw_gender.ne(customers_clean["gender"].astype("string")).sum()),
            "ages_out_of_range_set_missing": int(invalid_age.sum()),
            "missing_or_invalid_ages_filled_by_city_median": int(
                (numeric_age.isna() | invalid_age).sum()
            ),
            "signup_dates_converted": int(len(customers_clean)),
        },
        "products": {"rows_preserved_without_changes": int(len(products_raw))},
        "orders": {
            "timestamps_converted": int(len(orders_raw)),
            "exact_duplicate_rows_removed": exact_duplicate_rows,
            "older_duplicate_order_rows_removed": int(len(after_exact_dedup) - len(after_latest)),
            "unparseable_or_missing_amounts_filled_after_latest_record_selection": int(
                numeric_amount.isna().sum()
            ),
            "negative_delivery_times_set_missing_after_latest_record_selection": int(
                (negative_delivery & after_latest["status"].eq("completed")).sum()
            ),
            "missing_completed_delivery_times_filled_by_category_median": int(
                missing_completed_delivery.sum()
            ),
            "cancelled_delivery_times_left_missing": int(
                orders_clean.loc[orders_clean["status"].eq("cancelled"), "delivery_minutes"].isna().sum()
            ),
        },
    }


def save_clean_tables(
    customers: pd.DataFrame, products: pd.DataFrame, orders: pd.DataFrame
) -> None:
    CLEAN.mkdir(parents=True, exist_ok=True)
    customers.to_csv(CLEAN / "customers.csv", index=False, date_format="%Y-%m-%d")
    products.to_csv(CLEAN / "products.csv", index=False)
    orders.to_csv(CLEAN / "orders.csv", index=False)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as connection:
        customers_db = customers.copy()
        customers_db["signup_date"] = customers_db["signup_date"].dt.strftime("%Y-%m-%d")
        customers_db.to_sql("customers", connection, if_exists="replace", index=False)
        products.to_sql("products", connection, if_exists="replace", index=False)
        orders.to_sql("orders", connection, if_exists="replace", index=False)


def load_queries() -> list[str]:
    text = SQL_PATH.read_text(encoding="utf-8")
    text = re.sub(r"(?m)^\s*--[^\n]*(?:\n|$)", "", text)
    return [statement.strip() for statement in text.split(";") if statement.strip()]


def execute_queries() -> list[pd.DataFrame]:
    with sqlite3.connect(DB_PATH) as connection:
        return [pd.read_sql_query(query, connection) for query in load_queries()]


def analysis_report(query_results: list[pd.DataFrame], orders: pd.DataFrame) -> dict:
    completed = orders.loc[orders["status"].eq("completed")].copy()
    products = pd.read_csv(CLEAN / "products.csv")
    products = products.assign(price_inr=pd.to_numeric(products["price_inr"]))
    completed = completed.merge(
        products[["product_id", "category"]], on="product_id", how="left", validate="many_to_one"
    )

    pivot = pd.pivot_table(
        completed,
        values="order_amount",
        index="city",
        columns="category",
        aggfunc="sum",
        fill_value=0,
        margins=True,
        margins_name="All",
    ) if "city" in completed.columns else None
    # The Q2 cross-check needs the customer city, so join it from the clean table.
    clean_customers = pd.read_csv(CLEAN / "customers.csv")
    completed_city = completed.merge(
        clean_customers[["customer_id", "city"]], on="customer_id", how="left", validate="many_to_one"
    )
    pivot = pd.pivot_table(
        completed_city,
        values="order_amount",
        index="city",
        columns="category",
        aggfunc="sum",
        fill_value=0,
        margins=True,
        margins_name="All",
    )
    city_stats = completed_city.groupby("city", as_index=False).agg(
        sum_order_amount=("order_amount", "sum"),
        mean_order_amount=("order_amount", "mean"),
        order_count=("order_amount", "count"),
    )

    values = completed_city["order_amount"]
    q1, median, q3 = values.quantile([0.25, 0.5, 0.75]).tolist()
    iqr = q3 - q1
    lower_fence, upper_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    correlations = completed_city[
        ["order_amount", "discount_pct", "quantity", "delivery_minutes"]
    ].corr()
    pairs = [
        (left, right, float(correlations.loc[left, right]))
        for i, left in enumerate(correlations.columns)
        for right in correlations.columns[i + 1 :]
    ]
    strongest = max(pairs, key=lambda item: abs(item[2]))

    return {
        "query_rows": [int(len(result)) for result in query_results],
        "query_2_total_inr": round(float(query_results[1]["revenue_inr"].sum()), 2),
        "query_2_largest_cell": query_results[1].sort_values(
            "revenue_inr", ascending=False
        ).iloc[0].to_dict(),
        "query_5_average_spend_inr": round(float(query_results[4]["average_spend_inr"].iloc[0]), 2),
        "query_6_top_per_city": query_results[5].sort_values(["city", "row_num"])
        .groupby("city", as_index=False).first()[["city", "customer_id", "total_spend_inr"]]
        .to_dict(orient="records"),
        "query_6_rank_diff_row_number": int(
            query_results[5]["spend_rank"].ne(query_results[5]["row_num"]).sum()
        ),
        "query_7_daily_rows": int(len(query_results[6])),
        "query_7_last_cumulative_inr": round(float(query_results[6]["cumulative_revenue_inr"].iloc[-1]), 2),
        "q2_pivot": {
            "index": [str(index) for index in pivot.index],
            "columns": [str(column) for column in pivot.columns],
            "values": [[round(float(value), 2) for value in row] for row in pivot.to_numpy()],
            "grand_total_inr": round(float(pivot.loc["All", "All"]), 2),
        },
        "city_stats": city_stats.round({"sum_order_amount": 2, "mean_order_amount": 2}).to_dict(orient="records"),
        "amount_distribution": {
            "mean_inr": float(values.mean()),
            "median_inr": float(median),
            "skewness": float(values.skew()),
            "q1_inr": float(q1),
            "q2_inr": float(median),
            "q3_inr": float(q3),
            "lower_fence_inr": float(lower_fence),
            "upper_fence_inr": float(upper_fence),
            "outliers_below": int(values.lt(lower_fence).sum()),
            "outliers_above": int(values.gt(upper_fence).sum()),
        },
        "strongest_correlation_pair": {
            "left": strongest[0],
            "right": strongest[1],
            "correlation": strongest[2],
        },
        "correlation_matrix": correlations.round(6).to_dict(),
    }


def main() -> None:
    customers_raw, products_raw, orders_raw = load_raw()
    audit = raw_audit(customers_raw, products_raw, orders_raw)
    customers_clean = clean_customers(customers_raw)
    products_clean = products_raw.copy()
    orders_clean = clean_orders(orders_raw, products_clean)
    save_clean_tables(customers_clean, products_clean, orders_clean)
    results = execute_queries()
    report = {
        "raw_audit": audit,
        "clean_customers": {
            "city_counts": customers_clean["city"].value_counts().sort_index().to_dict(),
            "gender_counts": customers_clean["gender"].value_counts().sort_index().to_dict(),
            "missing_ages": int(customers_clean["age"].isna().sum()),
            "age_min": float(customers_clean["age"].min()),
            "age_max": float(customers_clean["age"].max()),
        },
        "clean_orders": {
            "rows": int(len(orders_clean)),
            "status_counts": orders_clean["status"].value_counts().sort_index().to_dict(),
            "pending_rows": int(orders_clean["status"].eq("pending").sum()),
            "missing_amounts": int(orders_clean["order_amount"].isna().sum()),
            "missing_completed_delivery": int(
                orders_clean.loc[orders_clean["status"].eq("completed"), "delivery_minutes"].isna().sum()
            ),
            "missing_cancelled_delivery": int(
                orders_clean.loc[orders_clean["status"].eq("cancelled"), "delivery_minutes"].isna().sum()
            ),
            "completed_revenue_inr": round(
                float(orders_clean.loc[orders_clean["status"].eq("completed"), "order_amount"].sum()), 2
            ),
        },
        "cleaning_changes": cleaning_changes(
            customers_raw, customers_clean, products_raw, orders_raw, orders_clean
        ),
        "analysis": analysis_report(results, orders_clean),
    }
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "part_a_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
