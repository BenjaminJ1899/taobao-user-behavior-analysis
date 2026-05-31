"""
生成模拟淘宝用户行为数据集

字段结构对齐阿里天池 UserBehavior 数据集：
  user_id        — 用户 ID（脱敏整数）
  item_id        — 商品 ID（脱敏整数）
  category_id    — 商品类目 ID（脱敏整数）
  behavior_type  — 行为类型：pv / fav / cart / buy
  time            — 行为时间（yyyy-mm-dd HH:MM:SS）

分布参数基于社区公开统计，确保业务逻辑合理：
  - 行为占比：pv≈80%, fav≈5%, cart≈10%, buy≈5%
  - 用户活跃度呈长尾分布（少数超级用户贡献大量行为）
  - 时间跨度 14 天，含昼夜/周末模式
"""

import pandas as pd
import numpy as np

# ---- 参数 ----
N = 200_000  # 总行为记录数
N_USERS = 5000
N_ITEMS = 8000
N_CATEGORIES = 50
START_DATE = "2017-11-25"
END_DATE = "2017-12-08"
OUTPUT = "data/user_behavior.csv"
SEED = 42

rng = np.random.default_rng(SEED)

# ---- 用户活跃度：幂律分布（少量超级用户，大量轻度用户）----
user_weights = rng.pareto(1.5, size=N_USERS)
user_weights /= user_weights.sum()
user_ids = rng.choice(np.arange(1, N_USERS + 1), size=N, p=user_weights)

# ---- 商品热度：同样幂律（少量爆款被大量浏览）----
item_weights = rng.pareto(1.3, size=N_ITEMS)
item_weights /= item_weights.sum()
item_ids = rng.choice(np.arange(1, N_ITEMS + 1), size=N, p=item_weights)

# ---- 类目（随机分配，类目大小不均）----
category_weights = rng.pareto(1.0, size=N_CATEGORIES)
category_weights /= category_weights.sum()
category_ids = rng.choice(np.arange(1, N_CATEGORIES + 1), size=N, p=category_weights)

# ---- 行为类型：pv 80%, cart 10%, fav 5%, buy 5% ----
behavior_types = rng.choice(
    ["pv", "cart", "fav", "buy"],
    size=N,
    p=[0.80, 0.10, 0.05, 0.05],
)

# ---- 时间戳：14 天内，含昼夜和周末模式 ----
dates = pd.date_range(START_DATE, END_DATE, freq="1min")
# 白天权重更高（9-23 点），模拟电商流量
hours = dates.hour
day_weights = np.where((hours >= 9) & (hours <= 23), 3.0, 1.0)
# 周末权重略高
day_of_week = dates.dayofweek
weekend_weights = np.where(day_of_week >= 5, 1.5, 1.0)
time_weights = day_weights * weekend_weights
time_weights /= time_weights.sum()

sampled_times = rng.choice(dates, size=N, p=time_weights, replace=True)
# 随机打散到秒级
random_seconds = rng.integers(0, 60, size=N)
sampled_times = sampled_times + pd.to_timedelta(random_seconds, unit="s")
sampled_times = sorted(sampled_times)

# ---- 组装 ----
df = pd.DataFrame(
    {
        "user_id": user_ids,
        "item_id": item_ids,
        "category_id": category_ids,
        "behavior_type": behavior_types,
        "time": sampled_times,
    }
)

# 去除完全重复行
df = df.drop_duplicates()

df.to_csv(OUTPUT, index=False)
print(f"生成完成: {len(df)} 行 → {OUTPUT}")
print(f"\n行为分布:\n{df['behavior_type'].value_counts(normalize=True)}")
print(f"\n用户数: {df['user_id'].nunique()}")
print(f"商品数: {df['item_id'].nunique()}")
print(f"时间范围: {df['time'].min()} ~ {df['time'].max()}")
print(f"\n前 5 行:\n{df.head()}")
