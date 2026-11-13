"""共用資料讀取與整合流程。所有路徑均以本檔位置為基準，方便本機與部署使用。"""
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR
PROCESSED_DIR = BASE_DIR


def _read_csv(name: str, required: set[str]) -> pd.DataFrame:
    """讀取 CSV 並在欄位不符時回報清楚的錯誤。"""
    path = RAW_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"找不到資料檔：{path}")
    frame = pd.read_csv(path)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} 缺少必要欄位：{', '.join(sorted(missing))}")
    return frame


@st.cache_data(show_spinner="正在整合資料…")
def load_integrated_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """回傳明細、訂單層級與顧客層級資料，保留缺少維度的明細。"""
    orders = _read_csv("orders.csv", {"order_id", "customer_id", "order_date", "status", "payment_type"})
    items = _read_csv("order_items.csv", {"order_id", "product_id", "quantity", "unit_price", "discount_rate"})
    products = _read_csv("products.csv", {"product_id", "category"})
    customers = _read_csv("customers.csv", {"customer_id", "signup_date", "acquisition_channel", "city", "segment"})

    orders["order_date"] = pd.to_datetime(orders["order_date"], errors="coerce")
    customers["signup_date"] = pd.to_datetime(customers["signup_date"], errors="coerce")
    detail = (items.merge(products[["product_id", "category"]], on="product_id", how="left", validate="many_to_one")
              .merge(orders, on="order_id", how="left", validate="many_to_one")
              .merge(customers, on="customer_id", how="left", validate="many_to_one"))
    detail["revenue"] = detail["quantity"] * detail["unit_price"] * (1 - detail["discount_rate"])
    detail["gross_amount"] = detail["quantity"] * detail["unit_price"]
    detail["category"] = detail["category"].fillna("未分類")
    detail["segment"] = detail["segment"].fillna("未分類")

    valid = detail.loc[detail["order_id"].notna()]
    order_features = (valid.groupby("order_id", as_index=False)
        .agg(customer_id=("customer_id", "first"), order_date=("order_date", "first"),
             status=("status", "first"), payment_type=("payment_type", "first"),
             segment=("segment", "first"), city=("city", "first"),
             acquisition_channel=("acquisition_channel", "first"),
             order_amount=("revenue", "sum"), gross_amount=("gross_amount", "sum"),
             item_count=("product_id", "size"),
             quantity_sum=("quantity", "sum"), product_count=("product_id", "nunique"),
             category_count=("category", "nunique"), avg_discount=("discount_rate", "mean")))
    customer_features = (customers.copy().merge(
        valid.loc[valid["status"].eq("completed")].groupby("customer_id")
        .agg(completed_orders=("order_id", "nunique"), total_spend=("revenue", "sum"),
             total_items=("quantity", "sum"), avg_order_value=("revenue", "sum")),
        on="customer_id", how="left"))
    # 平均訂單金額需先按訂單彙總，避免把明細筆數當成訂單數。
    order_stats = order_features.loc[order_features["status"].eq("completed")].groupby("customer_id")["order_amount"].mean()
    customer_features["avg_order_value"] = customer_features["customer_id"].map(order_stats)
    customer_features[["completed_orders", "total_spend", "total_items", "avg_order_value"]] = customer_features[
        ["completed_orders", "total_spend", "total_items", "avg_order_value"]].fillna(0)
    # vip 是分類任務的目標欄位 y；segment 原始標籤保留供資料檢視。
    customer_features["is_vip"] = customer_features["segment"].eq("vip").astype(int)
    return detail, order_features, customer_features


def export_integrated_data() -> tuple[Path, Path, Path]:
    """產生可重複建立的整合明細、訂單彙總與顧客特徵 CSV。"""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    detail, orders, customers = load_integrated_data()
    detail_path = PROCESSED_DIR / "integrated_sales.csv"
    orders_path = PROCESSED_DIR / "order_features.csv"
    customers_path = PROCESSED_DIR / "customer_features.csv"
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    orders.to_csv(orders_path, index=False, encoding="utf-8-sig")
    customers.to_csv(customers_path, index=False, encoding="utf-8-sig")
    return detail_path, orders_path, customers_path
