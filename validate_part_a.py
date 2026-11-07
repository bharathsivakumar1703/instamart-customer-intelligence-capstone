"""Assert the required Part A customer and order cleaning checkpoints."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
CUSTOMERS_PATH = ROOT / "data" / "clean" / "customers.csv"
ORDERS_PATH = ROOT / "data" / "clean" / "orders.csv"
DB_PATH = ROOT / "data" / "instamart.db"


def main() -> None:
    customers = pd.read_csv(CUSTOMERS_PATH)
    orders = pd.read_csv(ORDERS_PATH)

    expected_cities = {
        "Pune": 227,
        "Delhi": 210,
        "Bengaluru": 197,
        "Chennai": 191,
        "Hyderabad": 191,
        "Mumbai": 184,
    }
    expected_genders = {"female": 610, "male": 560, "unknown": 30}
    assert customers["city"].value_counts().to_dict() == expected_cities
    assert customers["gender"].value_counts().to_dict() == expected_genders
    assert customers["age"].isna().sum() == 0
    assert customers["age"].between(18, 60).all()

    assert len(orders) == 12309
    assert orders["order_id"].nunique() == 12309
    assert orders["status"].value_counts().to_dict() == {
        "completed": 11110,
        "cancelled": 1199,
    }
    assert not orders["status"].eq("pending").any()
    assert orders["order_amount"].isna().sum() == 0
    assert orders.loc[orders["status"].eq("completed"), "delivery_minutes"].isna().sum() == 0
    assert orders.loc[orders["status"].eq("cancelled"), "delivery_minutes"].isna().sum() == 1199
    completed_revenue = orders.loc[orders["status"].eq("completed"), "order_amount"].sum()
    assert abs(float(completed_revenue) - 2_998_668.80) < 0.01

    with sqlite3.connect(DB_PATH) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert {"customers", "products", "orders"}.issubset(tables)

    print("ALL CHECKPOINTS PASSED")


if __name__ == "__main__":
    main()
