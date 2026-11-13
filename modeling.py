"""模型共用欄位定義及訂單預測的時間特徵工程。"""
import pandas as pd

VIP_FEATURES = ["acquisition_channel", "city", "tenure_days", "completed_orders",
                "total_spend", "total_items", "avg_order_value"]
VIP_CATEGORICAL = ["acquisition_channel", "city"]
VIP_NUMERIC = [field for field in VIP_FEATURES if field not in VIP_CATEGORICAL]

ORDER_NUMERIC = ["gross_amount", "item_count", "quantity_sum", "product_count", "category_count", "avg_discount",
                 "prior_completed_orders", "prior_total_spend", "prior_avg_order_value",
                 "customer_tenure_days", "order_month", "order_weekday"]
ORDER_CATEGORICAL = ["payment_type", "city", "acquisition_channel"]
ORDER_FEATURES = ORDER_NUMERIC + ORDER_CATEGORICAL


def make_order_training_frame(orders: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    """建構訂單預測訓練資料；顧客消費特徵只使用目標訂單之前的紀錄。"""
    frame = orders.loc[orders["status"].eq("completed")].copy()
    frame["order_date"] = pd.to_datetime(frame["order_date"], errors="coerce")
    frame = frame.dropna(subset=["order_date"]).sort_values(["customer_id", "order_date", "order_id"]).copy()
    history = frame.groupby("customer_id", sort=False)
    frame["prior_completed_orders"] = history.cumcount()
    frame["prior_total_spend"] = history["order_amount"].transform(lambda values: values.cumsum().shift(1))
    prior_count = frame["prior_completed_orders"].where(frame["prior_completed_orders"].gt(0))
    frame["prior_avg_order_value"] = frame["prior_total_spend"] / prior_count
    info = customers[["customer_id", "signup_date", "city", "acquisition_channel"]].drop_duplicates("customer_id")
    frame = frame.merge(info, on="customer_id", how="left", validate="many_to_one", suffixes=("", "_customer"))
    frame["city"] = frame["city"].fillna(frame["city_customer"] if "city_customer" in frame else "未知")
    frame["acquisition_channel"] = frame["acquisition_channel"].fillna(
        frame["acquisition_channel_customer"] if "acquisition_channel_customer" in frame else "未知")
    frame["signup_date"] = pd.to_datetime(frame["signup_date"], errors="coerce")
    frame["customer_tenure_days"] = (frame["order_date"] - frame["signup_date"]).dt.days.clip(lower=0)
    frame["order_month"] = frame["order_date"].dt.month
    frame["order_weekday"] = frame["order_date"].dt.dayofweek
    return frame
