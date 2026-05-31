"""共用 SQLite 连接与查询辅助模块"""

import os
import sqlite3
import pandas as pd

# 自动定位项目根目录（向上两级：utils/ → 项目根）
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
DB_PATH = os.path.join(_PROJECT_ROOT, "data", "taobao_behavior.db")


def get_connection():
    """返回 SQLite 连接对象"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def load_csv_to_sqlite(csv_path, table_name):
    """将 CSV 文件导入 SQLite"""
    conn = get_connection()
    chunks = pd.read_csv(csv_path, chunksize=100000)
    for i, chunk in enumerate(chunks):
        chunk.to_sql(table_name, conn, if_exists="replace" if i == 0 else "append", index=False)
    conn.commit()
    conn.close()
    print(f"已导入表: {table_name}")


def query(sql):
    """执行 SQL 查询，返回 DataFrame"""
    conn = get_connection()
    df = pd.read_sql_query(sql, conn)
    conn.close()
    return df
