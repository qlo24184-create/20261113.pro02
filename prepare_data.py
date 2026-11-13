"""整合原始 CSV：執行 `python prepare_data.py` 會輸出至 data/processed。"""
from data_utils import export_integrated_data

if __name__ == "__main__":
    for output in export_integrated_data():
        print(f"已輸出：{output}")
