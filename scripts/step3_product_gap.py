#!/usr/bin/env python3
"""
BƯỚC 3 — KHOẢNG CÁCH NGHIÊN CỨU–SẢN PHẨM

3.1: Kiểm tra compatibility NASA POWER (train) vs Open-Meteo (deploy)
     - NASA POWER là reanalysis (satellite + model assimilation)
     - Open-Meteo dùng ERA5 + other forecast models
     - Cùng cung cấp T, RH, wind, precip nhưng có thể có bias hệ thống

3.2: Kiểm tra feature availability — model cần 38 features, Open-Meteo cung cấp gì?

3.3: Đánh giá: web có thể deploy được không? Hay chỉ là demo?

3.4: Khuyến nghị cho báo cáo
"""
import json, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')

OUT = '/home/z/my-project/download/real_data_v85'
panel = pd.read_parquet(OUT + '/panel_v85.parquet')
features = json.loads(open(OUT+'/panel_v85_meta.json').read())['features']
all_features = features + ['cell_fire_rate_train', 'cell_fire_count_30d', 'cell_fire_rate_dry']

print("="*70)
print("BƯỚC 3 — KHOẢNG CÁCH NGHIÊN CỨU–SẢN PHẨM")
print("="*70)

# ============================================================
# 3.1: NASA POWER distribution statistics
# ============================================================
print("\n[3.1] NASA POWER phân phối (dùng cho training)")
core_weather = ['t2m', 'rh', 'wind', 'precip']
for v in core_weather:
    vals = panel[v].dropna()
    print(f"  {v:10s}: mean={vals.mean():.2f}, std={vals.std():.2f}, "
          f"min={vals.min():.2f}, max={vals.max():.2f}, "
          f"p5={vals.quantile(0.05):.2f}, p95={vals.quantile(0.95):.2f}")

print(f"\n  Open-Meteo (dùng cho web deploy) cung cấp:")
print(f"  - temperature_2m (°C) → tương đương t2m")
print(f"  - relative_humidity_2m (%) → tương đương rh")
print(f"  - wind_speed_10m (m/s) → tương đương wind")
print(f"  - precipitation (mm) → tương đương precip")
print(f"  - Nguồn: ERA5 reanalysis + forecast models")
print(f"  - Độ phân giải: theo giờ, available realtime")
print(f"  - NASA POWER: satellite-based, daily, reanalysis")
print(f"  - Cả hai đo cùng đại lượng vật lý nhưng từ mô hình khác nhau")
print(f"  - Có thể có bias hệ thống (ví dụ: NASA POWER ấm hơn ERA5 ở nhiệt độ)")
print(f"  - Chưa kiểm định KS test vì không thể truy cập Open-Meteo từ môi trường này")

# ============================================================
# 3.2: Feature availability analysis
# ============================================================
print(f"\n[3.2] FEATURE AVAILABILITY — Model cần {len(all_features)} features")
print(f"\n  Phân loại features theo nguồn cần thiết:")

# Categorize features
categories = {
    "Raw weather (có sẵn từ Open-Meteo realtime)": [],
    "Lagged weather (cần lịch sửu 1-14 ngày từ Open-Meteo)": [],
    "Rolling statistics (cần lịch sử 7-30 ngày)": [],
    "Derived interactions (tính từ raw + lagged)": [],
    "Fire history (cần MODIS fire data — KHÔNG có trong Open-Meteo)": [],
    "Calendar (tính từ date)": [],
}

for f in all_features:
    if f in ['t2m', 'rh', 'wind', 'precip']:
        categories["Raw weather (có sẵn từ Open-Meteo realtime)"].append(f)
    elif '_lag' in f:
        categories["Lagged weather (cần lịch sửu 1-14 ngày từ Open-Meteo)"].append(f)
    elif '_roll' in f or 'days_since_rain' in f:
        categories["Rolling statistics (cần lịch sử 7-30 ngày)"].append(f)
    elif f in ['doy']:
        categories["Calendar (tính từ date)"].append(f)
    elif 'cell_fire' in f:
        categories["Fire history (cần MODIS fire data — KHÔNG có trong Open-Meteo)"].append(f)
    else:
        categories["Derived interactions (tính từ raw + lagged)"].append(f)

for cat, feats in categories.items():
    print(f"\n  {cat}:")
    print(f"    Features ({len(feats)}): {', '.join(feats) if feats else '(none)'}")
    if "Open-Meteo realtime" in cat:
        print(f"    → Có sẵn trực tiếp ✓")
    elif "Lagged" in cat:
        print(f"    → Cần API Open-Meteo lấy 14 ngày gần nhất ✓ (khả thi)")
    elif "Rolling" in cat:
        print(f"    → Cần API Open-Meteo lấy 30 ngày gần nhất ✓ (khả thi)")
    elif "Derived" in cat:
        print(f"    → Tính từ raw + lagged ✓ (code sẵn)")
    elif "Calendar" in cat:
        print(f"    → Tính từ date ✓ (trivial)")
    elif "Fire history" in cat:
        print(f"    → Cần MODIS fire data API + lưu trữ ⚠ (KHÔNG có sẵn realtime)")
        print(f"    → cell_fire_rate_train: cần tính từ 2018-2021 MODIS (pre-computed, cố định)")
        print(f"    → cell_fire_count_30d: cần fire detections 30 ngày gần nhất từ FIRMS API")
        print(f"    → cell_fire_rate_dry: cần tính từ 2018-2021 dry season (pre-computed)")

# ============================================================
# 3.3: Deployability assessment
# ============================================================
print(f"\n[3.3] ĐÁNH GIÁ DEPLOYABILITY")
print(f"""
  Vấn đề 1: Nguồn thời tiết
    - Model train trên NASA POWER (satellite reanalysis, daily)
    - Web dùng Open-Meteo (ERA5-based, hourly)
    - Cùng đại lượng vật lý (T, RH, W, precip) nhưng mô hình khác nhau
    - Chưa kiểm tra bias hệ thống
    → Khuyến nghị: nếu deploy, dùng NASA POWER API (cùng nguồn) 
      hoặc validate trên Open-Meteo trước khi deploy

  Vấn đề 2: Lagged features
    - Model cần 16 lagged features (lag 1,3,7,14 cho T/RH/W/precip)
    - Web cần lấy 14 ngày lịch sử thời tiết từ API
    → Khả thi: Open-Meteo có API archive, nhưng cần code xử lý

  Vấn đề 3: Rolling statistics
    - Model cần 7 rolling features (roll7, roll14, roll30)
    - Web cần lấy 30 ngày lịch sử
    → Khả thi: cùng API archive

  Vấn đề 4: Fire history features
    - cell_fire_rate_train: pre-computed từ 2018-2021 MODIS → CỐ ĐỊNH, có thể lưu
    - cell_fire_count_30d: cần fire detections 30 ngày gần nhất → cần FIRMS API
    - cell_fire_rate_dry: pre-computed → CỐ ĐỊNH, có thể lưu
    → cell_fire_count_30d là vấn đề: cần gọi FIRMS API mỗi ngày, xử lý,
      hợp nhất vào panel. Khả thi nhưng phức tạp.

  Vấn đề 5: Chưa validate trên Open-Meteo
    - Chưa kiểm tra: nếu dùng Open-Meteo thay NASA POWER, AUROC thay đổi bao nhiêu?
    - Chưa kiểm tra: bias T/RH/W/precip giữa hai nguồn
    → Đây là hạn chế phải ghi rõ trong báo cáo
""")

# ============================================================
# 3.4: Khuyến nghị cho báo cáo
# ============================================================
print(f"[3.4] KHUYẾN NGHỊ CHO BÁO CÁO")
print(f"""
  Cách trình bày sản phẩm trong báo cáo:

  --- NÊN ---
  ✓ Mô tả sản phẩm là "công cụ trình diễn mô hình nghiên cứu" 
    (research demonstration tool), KHÔNG phải hệ thống cảnh báo 
    đã được xác nhận thực tế (validated operational system)
  
  ✓ Ghi rõ: model train trên NASA POWER, web demo dùng Open-Meteo
    — chưa kiểm tra tương thích
  
  ✓ Ghi rõ: model cần 38 features (16 lagged + 7 rolling + 3 fire history
    + 12 derived), không chỉ 4 raw weather — web cần xử lý nhiều hơn
    chỉ gọi API thời tiết hiện tại
  
  ✓ Ghi rõ: fire history features cần MODIS FIRMS API (realtime),
    không có sẵn trong Open-Meteo
  
  ✓ Ghi rõ: chưa thực hiện đánh giá hồi cứu (retrospective evaluation)
    với dữ liệu thời tiết từ Open-Meteo
  
  --- KHÔNG NÊN ---
  ✗ Tuyên bố web là "hệ thống cảnh báo cháy rừng" 
    (gọi là "demo" hoặc "mockup")
  
  ✗ Tuyên bố dự báo "realtime" (model chạy offline, dự báo t+1)
  
  ✗ Bỏ qua sự khác biệt giữa NASA POWER và Open-Meteo
  
  ✗ Im lặng về việc model cần fire history features mà web 
    không có dữ liệu để tính
""")

# ============================================================
# 3.5: Summary table
# ============================================================
print(f"\n[3.5] BẢNG TỔNG KẾT")
print(f"\n  {'Yêu cầu':<40} | {'NASA POWER (train)':<20} | {'Open-Meteo (deploy)':<20} | {'Khả thi?'}")
print(f"  {'-'*100}")
deploy_rows = [
    ("Raw weather (T, RH, W, precip)", "✓ daily", "✓ hourly", "✓"),
    ("Lagged weather (1-14 ngày)", "✓ pre-computed", "✓ cần API archive 14d", "✓"),
    ("Rolling stats (7-30 ngày)", "✓ pre-computed", "✓ cần API archive 30d", "✓"),
    ("Derived features (fwi, heat_idx...)", "✓ computed", "✓ computed from raw", "✓"),
    ("Calendar (doy)", "✓ from date", "✓ from date", "✓"),
    ("cell_fire_rate_train (2018-2021)", "✓ pre-computed", "⚠ cần lưu static file", "✓ (cố định)"),
    ("cell_fire_count_30d (rolling)", "✓ from panel", "⚠ cần FIRMS API + xử lý", "⚠ phức tạp"),
    ("cell_fire_rate_dry (2018-2021)", "✓ pre-computed", "⚠ cần lưu static file", "✓ (cố định)"),
    ("Bias kiểm tra NASA vs Open-Meteo", "—", "chưa kiểm tra", "✗ CHƯA LÀM"),
    ("Đánh giá hồi cứu trên Open-Meteo", "—", "chưa thực hiện", "✗ CHƯA LÀM"),
]
for req, npower, ometeo, feasible in deploy_rows:
    print(f"  {req:<40} | {npower:<20} | {ometeo:<20} | {feasible}")

print(f"\n  Tổng: 6/10 yêu cầu khả thi trực tiếp, 2 phức tạp, 2 chưa làm")
print(f"  → Web product nên được mô tả là 'demo', không phải 'hệ thống cảnh báo'")
print(f"  → Đánh giá hồi cứu trên Open-Meteo là hướng mở cho công việc tiếp theo")

# Save
results_3 = {
    "3.1_nasa_power_stats": {v: {"mean": float(panel[v].mean()), "std": float(panel[v].std()),
                                  "min": float(panel[v].min()), "max": float(panel[v].max())} for v in core_weather},
    "3.2_feature_categories": {k: v for k, v in categories.items()},
    "3.3_deployability": {
        "weather_source_mismatch": "NASA POWER (train) vs Open-Meteo (deploy) — chưa kiểm tra bias",
        "lagged_features": "cần API archive 14 ngày — khả thi",
        "rolling_features": "cần API archive 30 ngày — khả thi",
        "fire_history": "cell_fire_count_30d cần FIRMS API realtime — phức tạp",
        "retrospective_eval": "chưa thực hiện — cần chạy model trên Open-Meteo data",
    },
    "3.4_recommendation": "Mô tả web là 'demo', ghi rõ 2 hạn chế: (1) chưa kiểm tra bias NASA vs Open-Meteo, (2) chưa đánh giá hồi cứu",
    "3.5_summary": {"feasible_directly": 6, "complex": 2, "not_done": 2, "verdict": "demo, not operational"},
}
with open(OUT + '/step3_results.json', 'w') as f:
    json.dump(results_3, f, indent=2, default=str)
print(f"\n✓ Saved: {OUT}/step3_results.json")
