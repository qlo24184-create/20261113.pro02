"""頁面一：資料品質檢查與營運概況圖表。"""
import streamlit as st

from data_utils import load_integrated_data

st.set_page_config(page_title="資料狀態與視覺化", page_icon="📊", layout="wide")
st.title("資料狀態與視覺化")

try:
    detail, orders, customers = load_integrated_data()
except (FileNotFoundError, ValueError, KeyError) as exc:
    st.error(f"資料載入失敗：{exc}")
    st.stop()

raw_names = ["customers.csv", "orders.csv", "order_items.csv", "products.csv"]
columns = st.columns(4)
for column, name, frame in zip(columns, raw_names, [customers, orders, detail, detail[["product_id", "category"]].drop_duplicates()]):
    column.metric(name, f"{len(frame):,} 列")

st.subheader("資料品質")
quality = []
for name, frame in [("整合明細", detail), ("訂單彙總", orders), ("顧客資料", customers)]:
    quality.append({"資料表": name, "列數": len(frame), "欄數": len(frame.columns),
                    "缺值總數": int(frame.isna().sum().sum()), "重複列數": int(frame.duplicated().sum())})
st.dataframe(quality, hide_index=True, use_container_width=True)
if detail["order_id"].isna().any():
    st.warning(f"有 {int(detail['order_id'].isna().sum()):,} 筆明細無法連結到訂單，仍保留在整合明細供檢查。")

completed = orders.loc[orders["status"].eq("completed")].copy()
if completed.empty:
    st.warning("沒有已完成訂單可供分析。")
    st.stop()

st.subheader("已完成訂單概況")
a, b, c = st.columns(3)
a.metric("已完成訂單", f"{len(completed):,}")
b.metric("已完成營收", f"NT${completed['order_amount'].sum():,.0f}")
c.metric("平均訂單金額", f"NT${completed['order_amount'].mean():,.0f}")

left, right = st.columns(2)
with left:
    st.markdown("#### 每月營收")
    monthly = (completed.dropna(subset=["order_date"]).assign(月份=lambda x: x["order_date"].dt.to_period("M").astype(str))
               .groupby("月份", as_index=False)["order_amount"].sum().rename(columns={"order_amount": "營收"}))
    st.bar_chart(monthly.set_index("月份"), use_container_width=True)
with right:
    st.markdown("#### 客群營收")
    by_segment = (completed.groupby("segment", as_index=False)["order_amount"].sum()
                  .rename(columns={"segment": "客群", "order_amount": "營收"}).sort_values("營收", ascending=False))
    st.bar_chart(by_segment.set_index("客群"), use_container_width=True)

st.markdown("#### 訂單明細預覽")
st.dataframe(orders.head(20), hide_index=True, use_container_width=True)
st.download_button("下載整合明細 CSV", detail.to_csv(index=False).encode("utf-8-sig"),
                   "integrated_sales.csv", "text/csv")
