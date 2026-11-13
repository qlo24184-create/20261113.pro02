"""頁面二：載入預先訓練的分類器，使用中文欄位預測顧客 VIP 機率。"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from data_utils import BASE_DIR, load_integrated_data

st.set_page_config(page_title="VIP 機率預測", page_icon="🎯", layout="wide")
st.title("VIP 機率預測")
st.caption("選擇已訓練模型並輸入顧客資料，即可預測目前 VIP 標籤機率。資料沒有 VIP 升級歷程，因此結果不代表未來升級機率。")

@st.cache_resource
def load_model_bundle(path: str):
    """每個模型只載入一次，避免 Streamlit 重繪頁面時重複讀取檔案。"""
    return joblib.load(path)

_, _, customers = load_integrated_data()
model_names = {"羅吉斯迴歸": "vip_logistic.joblib", "隨機森林": "vip_random_forest.joblib"}
choice = st.selectbox("分類模型", list(model_names))
artifact_path = BASE_DIR / "models" / model_names[choice]
if not artifact_path.exists():
    st.error("尚未找到預先訓練的模型。請先在終端機執行 `python train_models.py`。")
    st.stop()

bundle = load_model_bundle(str(artifact_path))
channel_options = sorted(customers["acquisition_channel"].dropna().unique().tolist())
city_options = sorted(customers["city"].dropna().unique().tolist())
with st.form("vip_input"):
    st.subheader("顧客資料")
    st.caption("填妥下列欄位後，按下按鈕才會進行預測。")
    left, right = st.columns(2)
    with left:
        channel = st.selectbox("獲客管道", channel_options)
        city = st.selectbox("居住城市", city_options)
        tenure = st.number_input("顧客年資（天）", min_value=0, max_value=10000, value=500, step=30)
        completed_orders = st.number_input("已完成訂單數", min_value=0, max_value=10000, value=5, step=1)
    with right:
        total_spend = st.number_input("累計消費金額（NT$）", min_value=0.0, value=20000.0, step=1000.0)
        total_items = st.number_input("累計購買商品數", min_value=0, max_value=100000, value=10, step=1)
        avg_order_value = st.number_input("平均訂單金額（NT$）", min_value=0.0, value=3000.0, step=100.0)
    submitted = st.form_submit_button("資料輸入完成，開始預測 VIP")

if submitted:
    sample = pd.DataFrame([{
        "acquisition_channel": channel,
        "city": city,
        "tenure_days": int(tenure),
        "completed_orders": int(completed_orders),
        "total_spend": float(total_spend),
        "total_items": int(total_items),
        "avg_order_value": float(avg_order_value),
    }], columns=bundle["features"])
    probability = float(bundle["pipeline"].predict_proba(sample)[0, 1])
    predicted = int(bundle["pipeline"].predict(sample)[0])
    st.subheader("預測結果")
    st.metric("VIP 機率", f"{probability:.1%}")
    st.success("預測結果：VIP" if predicted else "預測結果：非 VIP")
