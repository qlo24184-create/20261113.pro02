# 電商資料分析與預測

## 功能

以 `data/raw/` 內的 customers、orders、order_items、products 四個 CSV 建立整合資料，提供三個 Streamlit 頁面：

1. **資料狀態與視覺化**：資料品質、完成訂單營收與客群趨勢。
2. **VIP 機率預測**：選擇預先訓練的羅吉斯迴歸或隨機森林，輸入繁體中文欄位後直接預測 VIP 分類機率。
3. **訂單金額預測**：選擇預先訓練的隨機森林或 XGBoost，輸入繁體中文欄位後直接預測訂單金額。

訂單金額以明細的 `quantity × unit_price × (1 - discount_rate)` 加總；只使用 `completed` 訂單訓練迴歸模型。

## 安裝與啟動

建議使用 Python 3.12 以上版本。於 `classCode` 目錄執行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python prepare_data.py
python train_models.py
streamlit run app.py
```

`prepare_data.py` 會在 `data/processed/` 產生 `integrated_sales.csv`（整合明細）、`order_features.csv`（訂單層級彙總）及 `customer_features.csv`（顧客特徵，`is_vip` 為分類目標 y）。`train_models.py` 使用現有資料預訓練四個模型，並將 joblib 檔案輸出至 `models/`。原始 CSV 更新後，請重新執行這兩個步驟。

## 分析解讀

目前 customers.csv 只有當下的 `segment`，沒有升級時間或歷史快照。VIP 頁因此是在分類目前的 VIP 標籤，顯示的機率不是未來升級機率，也不能推論因果關係。若要評估「未來變 VIP」，需取得按時間記錄的客群狀態，並按時間切分訓練與測試資料。

VIP 分類使用目前的 VIP 標籤作為 y，頁面輸出機率與分類結果，不顯示訓練評估指標。訂單金額模型使用歷史訂單訓練，顧客歷史消費特徵只統計目標訂單之前的已完成訂單，避免將目標訂單金額洩漏到輸入特徵。頁面輸出預測金額，不顯示訓練評估指標。
