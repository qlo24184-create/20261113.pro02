"""電商資料分析與預測入口：Streamlit 會自動載入 pages/ 下的三個頁面。"""
import streamlit as st

st.set_page_config(page_title="電商分析與預測", page_icon="🛍️", layout="wide")
st.title("電商分析與預測")
st.markdown("""
歡迎使用專案儀表板。請從左側選擇頁面：

1. **資料狀態與視覺化**：檢查原始資料、整合結果與營運概況。
2. **VIP 機率預測**：以顧客資料比較羅吉斯迴歸與隨機森林。
3. **訂單金額預測**：以隨機森林與 XGBoost 預測已完成訂單金額。

營收定義為商品數量 × 單價 × (1 − 折扣率)，不含運費、稅與退款。
""")
