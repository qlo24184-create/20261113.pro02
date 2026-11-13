"""頁面三：載入預先訓練的迴歸器，使用中文欄位預測訂單金額。"""
from pathlib import Path
import subprocess
import sys

import joblib
import pandas as pd
import streamlit as st

from data_utils import BASE_DIR, load_integrated_data

st.set_page_config(page_title="訂單金額預測", page_icon="💰", layout="wide")
st.title("訂單金額預測")
st.caption("選擇已訓練模型並輸入訂單與顧客資料，預測已完成訂單的商品金額總和（NT$）。")

@st.cache_resource
def load_model_bundle(path: str):
    """每個模型只載入一次，避免 Streamlit 重繪頁面時重複讀取檔案。"""
    return joblib.load(path)

_, orders, customers = load_integrated_data()
model_names = {"隨機森林": "order_random_forest.joblib", "XGBoost": "order_xgboost.joblib"}
choice = st.selectbox("迴歸模型", list(model_names))
artifact_path = BASE_DIR / "models" / model_names[choice]
if not artifact_path.exists():
    with st.spinner("首次使用正在雲端訓練預測模型，稍後約15秒..."):
         subprocess.run([sys.executable, "train_models.py"], cwd=BASE_DIR, check=True)
    

payment_options = sorted(orders["payment_type"].dropna().unique().tolist())
city_options = sorted(customers["city"].dropna().unique().tolist())
channel_options = sorted(customers["acquisition_channel"].dropna().unique().tolist())
weekday_options = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]

with st.form("order_input"):
    st.subheader("訂單資料")
    st.caption("填妥下列欄位後，按下按鈕才會進行預測。")
    left, middle, right = st.columns(3)
    with left:
        gross_amount = st.number_input("折扣前商品總額（NT$）", min_value=0.0, value=5000.0, step=500.0)
        item_count = st.number_input("商品明細筆數", min_value=1, max_value=1000, value=2, step=1)
        quantity_sum = st.number_input("商品總數量", min_value=1, max_value=10000, value=2, step=1)
        product_count = st.number_input("不同商品數", min_value=1, max_value=1000, value=2, step=1)
        category_count = st.number_input("商品類別數", min_value=1, max_value=100, value=1, step=1)
    with middle:
        avg_discount = st.slider("平均折扣率", min_value=0.0, max_value=0.9, value=0.05, step=0.01)
        payment_type = st.selectbox("付款方式", payment_options)
        order_month = st.selectbox("訂單月份", list(range(1, 13)), format_func=lambda month: f"{month} 月")
        weekday_label = st.selectbox("訂單星期", weekday_options)
        customer_tenure_days = st.number_input("顧客年資（天）", min_value=0, max_value=10000, value=500, step=30)
    with right:
        city = st.selectbox("顧客居住城市", city_options)
        acquisition_channel = st.selectbox("顧客獲客管道", channel_options)
        prior_completed_orders = st.number_input("顧客過去已完成訂單數", min_value=0, max_value=10000, value=3, step=1)
        prior_total_spend = st.number_input("顧客過去累計消費（NT$）", min_value=0.0, value=10000.0, step=1000.0)
        prior_avg_order_value = st.number_input("顧客過去平均訂單金額（NT$）", min_value=0.0, value=2500.0, step=100.0)
    submitted = st.form_submit_button("資料輸入完成，開始預測訂單金額")

if submitted:
    sample = pd.DataFrame([{
        "gross_amount": float(gross_amount),
        "item_count": int(item_count),
        "quantity_sum": int(quantity_sum),
        "product_count": int(product_count),
        "category_count": int(category_count),
        "avg_discount": float(avg_discount),
        "prior_completed_orders": int(prior_completed_orders),
        "prior_total_spend": float(prior_total_spend),
        "prior_avg_order_value": float(prior_avg_order_value),
        "customer_tenure_days": int(customer_tenure_days),
        "order_month": int(order_month),
        "order_weekday": weekday_options.index(weekday_label),
        "payment_type": payment_type,
        "city": city,
        "acquisition_channel": acquisition_channel,
    }])
    bundle = load_model_bundle(str(artifact_path))
    sample = sample[bundle["features"]]
    prediction = float(bundle["pipeline"].predict(sample)[0])
    st.subheader("預測結果")
    st.metric("預測訂單金額", f"NT${max(prediction, 0):,.0f}")
