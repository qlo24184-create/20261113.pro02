"""使用全部可用樣本預先訓練並匯出兩個分類器與兩個迴歸器。

執行：python train_models.py
模型存放於 models/，供 Streamlit 頁面直接載入與預測。
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from data_utils import BASE_DIR, load_integrated_data
from modeling import (ORDER_CATEGORICAL, ORDER_FEATURES, ORDER_NUMERIC,
                      VIP_CATEGORICAL, VIP_FEATURES, VIP_NUMERIC,
                      make_order_training_frame)


def make_transformer(numeric: list[str], categorical: list[str], scale: bool = False) -> ColumnTransformer:
    """依欄位型態填補缺值、編碼類別欄位，分類模型再縮放數值。"""
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), numeric),
        ("category", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])


def save_pipeline(name: str, pipeline: Pipeline, features: list[str], target: str, count: int) -> None:
    """將模型與輸入欄位定義一起序列化，確保頁面傳入欄位一致。"""
    output = BASE_DIR / "models" / f"{name}.joblib"
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "features": features, "target": target, "training_rows": count}, output, compress=3)
    print(f"已輸出 {output.relative_to(BASE_DIR)}（訓練資料 {count:,} 筆）")


def main() -> None:
    _, orders, customers = load_integrated_data()
    # 客戶目標 y：1 表示目前 segment 為 vip，0 表示其他客群。
    vip = customers.copy()
    vip["signup_date"] = pd.to_datetime(vip["signup_date"], errors="coerce")
    vip["tenure_days"] = (pd.Timestamp("2025-12-31") - vip["signup_date"]).dt.days.clip(lower=0)
    vip["is_vip"] = vip["segment"].eq("vip").astype(int)
    for name, estimator in [
        ("vip_logistic", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42)),
        ("vip_random_forest", RandomForestClassifier(n_estimators=350, min_samples_leaf=2,
                                                       class_weight="balanced", random_state=42, n_jobs=-1)),
    ]:
        pipeline = Pipeline([("features", make_transformer(VIP_NUMERIC, VIP_CATEGORICAL, scale=True)),
                             ("model", estimator)])
        pipeline.fit(vip[VIP_FEATURES], vip["is_vip"])
        save_pipeline(name, pipeline, VIP_FEATURES, "is_vip", len(vip))

    regression = make_order_training_frame(orders, customers).dropna(subset=["order_amount"])
    regressors = [
        ("order_random_forest", RandomForestRegressor(n_estimators=350, min_samples_leaf=2,
                                                        random_state=42, n_jobs=-1)),
        ("order_xgboost", XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.04,
                                        subsample=0.85, colsample_bytree=0.85,
                                        objective="reg:squarederror", random_state=42, n_jobs=-1)),
    ]
    for name, estimator in regressors:
        pipeline = Pipeline([("features", make_transformer(ORDER_NUMERIC, ORDER_CATEGORICAL)),
                             ("model", estimator)])
        pipeline.fit(regression[ORDER_FEATURES], regression["order_amount"])
        save_pipeline(name, pipeline, ORDER_FEATURES, "order_amount", len(regression))


if __name__ == "__main__":
    main()
