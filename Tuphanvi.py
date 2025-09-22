# grouped_stats_simple.py
# Chạy: streamlit run grouped_stats_simple.py

import math
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Grouped Data Stats (No Charts)", layout="centered")
st.title("Tính bảng số liệu ghép nhóm.")
st.caption("Nhập [cận dưới, cận trên) và tần số. Tính Q1/Q2/Q3, trung bình/phương sai/độ lệch chuẩn (theo trung điểm), và mốt.")

# ----------------- helper functions -----------------

def validate_classes(df: pd.DataFrame) -> Tuple[bool, str]:
    """Kiểm tra lớp: cận dưới < cận trên, tăng dần, không chồng lấn, tần số ≥ 0."""
    if df.empty:
        return False, "Bảng rỗng."
    required = {"cận dưới", "cận trên", "tần số"}
    if not required.issubset(df.columns):
        return False, f"Thiếu cột. Cần có: {', '.join(required)}"

    try:
        df["cận dưới"] = pd.to_numeric(df["cận dưới"])
        df["cận trên"] = pd.to_numeric(df["cận trên"])
        df["tần số"]  = pd.to_numeric(df["tần số"])
    except Exception:
        return False, "Không chuyển được sang số. Hãy kiểm tra dữ liệu."

    if (df["tần số"] < 0).any():
        return False, "Tần số phải ≥ 0."

    df.sort_values("cận dưới", inplace=True, ignore_index=True)

    for i, row in df.iterrows():
        if not (row["cận dưới"] < row["cận trên"]):
            return False, f"Hàng {i+1}: cận dưới phải < cận trên."
        if i > 0:
            prev = df.iloc[i-1]
            # lớp dạng [cận dưới, cận trên), không chồng nhau
            if row["cận dưới"] < prev["cận trên"] - 1e-12:
                return False, f"Lớp {i} và {i+1} chồng lấn (cận trên_{i} > cận dưới_{i+1})."

    return True, ""


def grouped_percentile(df: pd.DataFrame, p: float) -> float:
    """
    Percentile cho dữ liệu ghép nhóm (nội suy tuyến tính):
      k = p/100 * N
      P_p = L + ((k - CF_prev)/f) * h
    """
    N = df["tần số"].sum()
    if N <= 0:
        raise ValueError("Tổng tần số N phải > 0.")
    if p <= 0:
        return float(df["cận dưới"].iloc[0])
    if p >= 100:
        return float(df["cận trên"].iloc[-1])

    work = df.copy()
    work["cf"] = work["tần số"].cumsum()
    k = p / 100.0 * N

    idx = (work["cf"] >= k).idxmax()
    cf_prev = float(work["cf"].iloc[idx-1]) if idx > 0 else 0.0
    f_class = float(work["tần số"].iloc[idx])
    L = float(work["cận dưới"].iloc[idx])
    h = float(work["cận trên"].iloc[idx] - work["cận dưới"].iloc[idx])

    if f_class <= 0 or h <= 0:
        return L
    return L + ((k - cf_prev) / f_class) * h


def grouped_quartiles(df: pd.DataFrame):
    q1 = grouped_percentile(df, 25.0)
    q2 = grouped_percentile(df, 50.0)
    q3 = grouped_percentile(df, 75.0)
    return q1, q2, q3


def grouped_mean_variance(df: pd.DataFrame):
    """
    Trung bình/Phương sai/Độ lệch chuẩn xấp xỉ theo trung điểm lớp:
      mean ≈ sum(f*m)/N
      var ≈ sum(f*(m-mean)^2) / N
    """
    N = df["tần số"].sum()
    if N <= 0:
        return float("nan"), float("nan"), float("nan")

    m = (df["cận dưới"] + df["cận trên"]) / 2.0
    gmean = float((df["tần số"] * m).sum() / N)

    var_num = (df["tần số"] * (m - gmean) ** 2).sum()
    denom = N
    if denom <= 0:
        gvar = float("nan")
        gstd = float("nan")
    else:
        gvar = float(var_num / denom)
        gstd = float(math.sqrt(gvar))
    return gmean, gvar, gstd


def grouped_mode(df: pd.DataFrame) -> float:
    """
    Mốt (mode) dữ liệu ghép nhóm:
      Với lớp mốt có tần số f1, lớp trước f0, lớp sau f2, cận dưới L, độ rộng h:
      Mode ≈ L + ((f1 - f0) / (2*f1 - f0 - f2)) * h
      Nếu không có lớp trước/ sau thì giả f0=0 hoặc f2=0.
    """
    if df["tần số"].sum() <= 0:
        return float("nan")

    i = int(df["tần số"].idxmax())  # chỉ số lớp mốt
    f1 = float(df.loc[i, "tần số"])
    L  = float(df.loc[i, "cận dưới"])
    h  = float(df.loc[i, "cận trên"] - df.loc[i, "cận dưới"])
    f0 = float(df.loc[i-1, "tần số"]) if i - 1 >= 0 else 0.0
    f2 = float(df.loc[i+1, "tần số"]) if i + 1 < len(df) else 0.0

    denom = (2 * f1 - f0 - f2)
    if denom == 0 or h <= 0:
        return L
    return L + ((f1 - f0) / denom) * h


# ----------------- UI -----------------

st.sidebar.write("Bảng số liệu mẫu:")
st.sidebar.code("cận dưới,cận trên,tần số\n0,10,5\n10,20,8\n20,30,12\n30,40,7\n")

st.subheader("1) Nhập bảng lớp [cận dưới, cận trên) và tần số (tần số)")

df0 = pd.DataFrame(
    {"cận dưới": [0, 10, 20, 30],
     "cận trên": [10, 20, 30, 40],
     "tần số":  [5,  8,  12,  7]}
    )

df = st.data_editor(
    df0,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "cận dưới": st.column_config.NumberColumn("cận dưới"),
        "cận trên": st.column_config.NumberColumn("cận trên"),
        "tần số":  st.column_config.NumberColumn("tần số", min_value=0, step=1),
    },
)

ok, msg = validate_classes(df.copy())
if not ok:
    st.error(f"Lỗi dữ liệu: {msg}")
    st.stop()

df.sort_values("cận dưới", inplace=True, ignore_index=True)
df["width"] = df["cận trên"] - df["cận dưới"]
df["mid"] = (df["cận dưới"] + df["cận trên"]) / 2.0
df["cf"] = df["tần số"].cumsum()
N = int(df["tần số"].sum())

st.caption(f"Tổng tần số N = **{N}**")
if N == 0:
    st.info("Tổng tần số bằng 0. Hãy nhập tần số dương.")
    st.stop()

# ----------------- calculations -----------------

q1, q2, q3 = grouped_quartiles(df)
iqr = q3 - q1
gmean, gvar, gstd = grouped_mean_variance(df)
mode_val = grouped_mode(df)

# ----------------- results -----------------

st.subheader("2) Kết quả")
st.markdown(
    f"""
- **Q1** = **{q1:.6g}**  
- **Trung vị / Q2** = **{q2:.6g}**  
- **Q3** = **{q3:.6g}**  
- **Khoảng tứ phân vị** = Q3 − Q1 = **{iqr:.6g}**  
- **Mốt (mode)** ≈ **{mode_val:.6g}** *(công thức mốt)*  
- **Trung bình**≈ **{gmean:.6g}**  
- **Phương sai** ≈ **{gvar:.6g}**  
- **Độ lệch chuẩn** ≈ **{gstd:.6g}**
"""
)
