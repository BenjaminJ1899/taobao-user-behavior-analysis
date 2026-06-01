"""
生成模拟淘宝用户行为数据集（因果结构版）

核心改进（v2）：
  1. item_id → category_id 固定映射（每个商品只属于一个品类）
  2. 用户行为按「会话 → 商品 → 行为路径」因果链生成
  3. 同一用户-商品对内的行为有依赖关系（pv→fav→cart→buy 状态机）
  4. 用户在会话内的时间是连续的（不是独立随机时间戳）

字段结构对齐阿里天池 UserBehavior 数据集：
  user_id        — 用户 ID（脱敏整数）
  item_id        — 商品 ID（脱敏整数）
  category_id    — 商品类目 ID（脱敏整数，通过 item_id 查映射表）
  behavior_type  — 行为类型：pv / fav / cart / buy
  time           — 行为时间（yyyy-mm-dd HH:MM:SS）
"""

import pandas as pd
import numpy as np
from datetime import timedelta

# ---- 参数 ----
TARGET_ROWS = 200_000
N_USERS = 10000
N_ITEMS = 10000
N_CATEGORIES = 50
START_DATE = pd.Timestamp("2017-11-25")
END_DATE = pd.Timestamp("2017-12-08")
OUTPUT = "data/user_behavior.csv"
SEED = 42

rng = np.random.default_rng(SEED)

# ============================================================
# 1. 建立 item_id → category_id 固定映射
# ============================================================
# 品类大小不均（幂律），每个商品固定属于一个品类
cat_weights = rng.pareto(1.0, size=N_CATEGORIES)
cat_weights /= cat_weights.sum()
item_to_category = rng.choice(np.arange(1, N_CATEGORIES + 1), size=N_ITEMS, p=cat_weights)

# 商品热度：幂律（少量爆款被大量浏览）
item_popularity = rng.pareto(1.3, size=N_ITEMS)
item_popularity /= item_popularity.sum()

# ============================================================
# 2. 用户活跃度：幂律长尾
# ============================================================
user_activity = rng.pareto(1.5, size=N_USERS)
# 缩放到每个用户的会话数（1 ~ 80 个会话）
user_sessions = np.round(user_activity / user_activity.max() * 149 + 1).astype(int)

# ============================================================
# 3. 时间权重：昼夜 + 周末
# ============================================================
date_range = pd.date_range(START_DATE, END_DATE, freq="1h")
hours = date_range.hour
time_weights = np.where((hours >= 9) & (hours <= 23), 3.0, 1.0)
weekend = date_range.dayofweek >= 5
time_weights = time_weights * np.where(weekend, 1.5, 1.0)
time_weights /= time_weights.sum()

# ============================================================
# 4. 行为路径状态机
#    给定一个用户-商品交互，概率性地生成行为序列
# ============================================================
# 每个用户-商品对的行为概率
P_FAV_AFTER_PV = 0.05    # 浏览后有 5% 概率收藏
P_CART_AFTER_PV = 0.10   # 浏览后有 10% 概率加购
P_BUY_AFTER_PV = 0.01    # 浏览后有 1% 概率直接购买（闪电单）
P_CART_AFTER_FAV = 0.10  # 收藏后有 10% 概率加购
P_BUY_AFTER_CART = 0.15  # 加购后有 15% 概率购买
P_BUY_AFTER_FAV = 0.03   # 收藏后有 3% 概率购买（罕见）


def generate_behavior_path():
    """
    为一个用户-商品交互生成行为序列。
    返回 list，如 ['pv'] 或 ['pv', 'cart'] 或 ['pv', 'cart', 'buy']
    """
    path = ["pv"]

    # 浏览后 → 收藏 / 加购 / 直接买（可同时发生）
    did_fav = rng.random() < P_FAV_AFTER_PV
    did_cart = rng.random() < P_CART_AFTER_PV
    did_buy_direct = rng.random() < P_BUY_AFTER_PV

    if did_fav:
        path.append("fav")
    if did_cart:
        path.append("cart")
    if did_buy_direct:
        path.append("buy")
        return path  # 买了就不继续了

    # 收藏 → 加购
    if did_fav and not did_cart and rng.random() < P_CART_AFTER_FAV:
        path.append("cart")
        did_cart = True

    # 加购 → 购买
    if did_cart and rng.random() < P_BUY_AFTER_CART:
        path.append("buy")

    return path


# ============================================================
# 5. 生成数据：为每个用户的每个会话生成行为
#
#    关键：用户的会话在时间上聚类，而非随机散布在 14 天中。
#    每个用户的活跃日为连续日块（模拟用户集中使用行为），
#    留存分析因此有意义——高活跃用户连续多天出现。
# ============================================================
rows = []

all_dates = sorted(set(d.date() for d in date_range))
date_to_hours = {}
for d in all_dates:
    date_to_hours[d] = [h for h in date_range if h.date() == d]

for user_idx in range(N_USERS):
    uid = user_idx + 1
    n_sessions = user_sessions[user_idx]

    # 活跃天数：分级分布（~30% 轻度 / ~40% 中度 / ~30% 重度）
    n_active_days = int(rng.choice(
        [1, 2, 3, 4, 5, 7, 10, 14],
        p=[0.38, 0.17, 0.12, 0.10, 0.08, 0.07, 0.05, 0.03]
    ))
    # 随机选起始日，连续取 n_active_days 天
    max_start = max(0, len(all_dates) - n_active_days)
    start_idx = rng.integers(0, max(1, max_start + 1))
    active_days = all_dates[start_idx:start_idx + n_active_days]

    # 会话数与活跃天数正相关
    sessions_per_day = rng.integers(1, 3, size=n_active_days)  # 每天 1-2 个会话
    total_sessions = sessions_per_day.sum()

    for _ in range(total_sessions):
        # 选一个活跃日
        session_day = active_days[rng.integers(0, len(active_days))]
        day_hours = date_to_hours[session_day]
        if len(day_hours) == 0:
            continue
        hour_w = np.where(np.array([h.hour for h in day_hours]) >= 9, 3.0, 1.0)
        hour_w /= hour_w.sum()
        session_start = rng.choice(day_hours, p=hour_w)

        # 每个会话浏览 1~6 个商品
        n_items_in_session = rng.integers(1, 7)

        for item_order_in_session in range(n_items_in_session):
            # 按商品热度选商品
            iid = rng.choice(np.arange(1, N_ITEMS + 1), p=item_popularity)
            cid = item_to_category[iid - 1]

            # 生成行为路径
            path = generate_behavior_path()

            # 每个行为间隔 10 秒 ~ 5 分钟（路径内连续）
            offset_seconds = 0
            for behavior in path:
                event_time = session_start + timedelta(
                    seconds=int(offset_seconds + rng.integers(10, 300))
                )
                rows.append(
                    {
                        "user_id": uid,
                        "item_id": iid,
                        "category_id": cid,
                        "behavior_type": behavior,
                        "time": event_time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )
                offset_seconds += 60  # 同路径内行为间隔

# ============================================================
# 6. 组装、去重、保存
# ============================================================
df = pd.DataFrame(rows)

# 按时间排序
df = df.sort_values("time").reset_index(drop=True)

# 去重（同用户同商品同行为同秒视为重复采集）
df = df.drop_duplicates(subset=["user_id", "item_id", "behavior_type", "time"])

# 如果超过目标，按用户采样（保留每个用户的完整行为数据）
if len(df) > TARGET_ROWS:
    all_users = df["user_id"].unique()
    n_keep = int(len(all_users) * TARGET_ROWS / len(df))
    keep_users = rng.choice(all_users, size=max(n_keep, 100), replace=False)
    df = df[df["user_id"].isin(keep_users)].sort_values("time").reset_index(drop=True)

df.to_csv(OUTPUT, index=False)

# ---- 输出统计 ----
print(f"生成完成: {len(df):,} 行 → {OUTPUT}")
print(f"\n行为分布:\n{df['behavior_type'].value_counts(normalize=True).mul(100).round(2)}")
print(f"\n用户数: {df['user_id'].nunique():,}")
print(f"商品数: {df['item_id'].nunique():,}")
print(f"品类数: {df['category_id'].nunique()}")
print(f"时间范围: {df['time'].min()} ~ {df['time'].max()}")

# 验证因果结构
print(f"\n=== 因果结构验证 ===")
# 验证 item-category 映射
ic_check = df.groupby("item_id")["category_id"].nunique()
print(f"每个 item 只属于一个 category: {ic_check.max() == 1}")

# 验证用户路径
paths = (
    df.groupby(["user_id", "item_id"])["behavior_type"]
    .apply(lambda x: "→".join(sorted(x.unique(), key=lambda v: {"pv": 1, "fav": 2, "cart": 3, "buy": 4}[v])))
    .reset_index()
)
path_counts = paths["behavior_type"].value_counts().head(10)
print(f"\nTop 10 用户路径:")
for p, c in path_counts.items():
    print(f"  {p}: {c:,} ({c/len(paths)*100:.1f}%)")

print(f"\n前 10 行:")
print(df.head(10).to_string(index=False))
