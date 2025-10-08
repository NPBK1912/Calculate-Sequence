import re
import ast
import math
from typing import Dict, Tuple, List

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Tính dãy số", layout="wide")

# ---------- CSS ----------
st.markdown(
    """
<style>
section[data-testid="stSidebar"] .stButton > button {
  width: 100% !important; min-width: 0 !important; height: 40px !important;
  display: flex !important; align-items: center !important; justify-content: center !important;
  padding: 0 10px !important;
}
section[data-testid="stSidebar"] .stButton > button * {
  overflow: hidden !important; text-overflow: ellipsis !important; white-space: nowrap !important;
}
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] {
  display: flex !important; flex-wrap: wrap !important; gap: 8px !important;
}
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
  flex: 1 1 calc(25% - 8px) !important; max-width: calc(25% - 8px) !important;
}
@media (max-width: 280px) {
  section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
    flex: 1 1 calc(50% - 8px) !important; max-width: calc(50% - 8px) !important;
  }
}
@media (max-width: 200px) {
  section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
    flex: 1 1 100% !important; max-width: 100% !important;
  }
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------- Safe eval ----------
ALLOWED_MATH_NAMES = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "log": lambda x, base=None: math.log(x) if base is None else math.log(x, base),
    "ln": math.log,
    "sqrt": math.sqrt,
    "abs": abs,
    "floor": math.floor,
    "ceil": math.ceil,
    "pi": math.pi,
    "e": math.e,
}

SAFE_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Num,
    ast.Subscript,
    ast.Index,
    ast.Slice,
    ast.Tuple,
    ast.List,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.UAdd,
    ast.USub,
    ast.LShift,
    ast.RShift,
    ast.BitOr,
    ast.BitAnd,
    ast.BitXor,
)


def assert_safe_expr(expr: str):
    try:
        node = ast.parse(expr, mode="eval")
    except Exception as e:
        raise ValueError(f"Invalid expression: {e}")
    for n in ast.walk(node):
        if not isinstance(n, SAFE_NODES):
            raise ValueError(f"Unsafe or unsupported expression element: {type(n).__name__}")


def safe_eval(expr: str, names: Dict = None):
    names = names or {}
    assert_safe_expr(expr)
    code = compile(ast.parse(expr, mode="eval"), "<string>", "eval")
    env = {"__builtins__": None, **ALLOWED_MATH_NAMES, **names}
    return eval(code, env, {})


# ---------- LaTeX parsing ----------
def _find_match(s: str, i: int, open_ch: str, close_ch: str) -> int:
    if i < 0 or i >= len(s) or s[i] != open_ch:
        return -1
    depth, j = 1, i + 1
    while j < len(s):
        c = s[j]
        if c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return j
        j += 1
    return -1


def _token_left(s: str, pos: int) -> Tuple[int, str]:
    j = pos - 1
    while j >= 0 and s[j].isspace():
        j -= 1
    if j < 0:
        return (0, "")

    if s[j] in ")]}":
        pair = {")": "(", "]": "[", "}": "{"}
        opener = pair[s[j]]
        k, depth = j, 0
        while k >= 0:
            if s[k] == s[j]:
                depth += 1
            elif s[k] == opener:
                depth -= 1
                if depth == 0:
                    return k, s[k : j + 1]
            k -= 1
        return (0, s[: j + 1])

    if s[j] == "]":
        b, depth = j, 0
        while b >= 0:
            if s[b] == "]":
                depth += 1
            elif s[b] == "[":
                depth -= 1
                if depth == 0:
                    break
            b -= 1
        a = b - 1
        if a >= 0 and s[a] == "u":
            return a, s[a : j + 1]
        return b, s[b : j + 1]

    k = j
    while k >= 0 and re.match(r"[A-Za-z0-9_.]", s[k]):
        k -= 1
    return k + 1, s[k + 1 : j + 1]


def _replace_all_frac(s: str) -> str:
    i, out = 0, s
    while True:
        m = re.search(r"\\d?frac\s*\{", out[i:])
        if not m:
            break
        start = i + m.start()
        lb1 = start + m.group(0).rfind("{")
        rb1 = _find_match(out, lb1, "{", "}")
        if rb1 == -1:
            i = start + 1
            continue
        j = rb1 + 1
        while j < len(out) and out[j].isspace():
            j += 1
        if j >= len(out) or out[j] != "{":
            i = start + 1
            continue
        lb2 = j
        rb2 = _find_match(out, lb2, "{", "}")
        if rb2 == -1:
            i = start + 1
            continue
        num = out[lb1 + 1 : rb1]
        den = out[lb2 + 1 : rb2]
        repl = f"({num})/({den})"
        out = out[:start] + repl + out[rb2 + 1 :]
        i = start + len(repl)
    return out


def _replace_all_sqrt(s: str) -> str:
    i, out = 0, s
    while True:
        m = re.search(r"\\sqrt", out[i:])
        if not m:
            break
        start = i + m.start()
        j = start + len("\\sqrt")
        k_idx = None
        if j < len(out) and out[j] == "[":
            rb = _find_match(out, j, "[", "]")
            if rb == -1:
                i = start + 1
                continue
            k_idx = out[j + 1 : rb].strip()
            j = rb + 1
        while j < len(out) and out[j].isspace():
            j += 1
        if j >= len(out) or out[j] != "{":
            i = start + 1
            continue
        rb2 = _find_match(out, j, "{", "}")
        if rb2 == -1:
            i = start + 1
            continue
        expr = out[j + 1 : rb2]
        repl = f"sqrt({expr})" if not k_idx else f"(({expr})**(1.0/{k_idx}))"
        out = out[:start] + repl + out[rb2 + 1 :]
        i = start + len(repl)
    return out


def _replace_all_power(s: str) -> str:
    i, out = 0, s
    while i < len(out):
        if out[i] == "^":
            if i + 1 < len(out) and out[i + 1] == "{":
                rb = _find_match(out, i + 1, "{", "}")
                if rb == -1:
                    i += 1
                    continue
                exp_str = out[i + 2 : rb]
                base_start, base_tok = _token_left(out, i)
                if base_tok == "":
                    i = rb + 1
                    continue
                repl = f"({base_tok})**({exp_str})"
                out = out[:base_start] + repl + out[rb + 1 :]
                i = base_start + len(repl)
                continue
            else:
                j = i + 1
                if j < len(out) and out[j] in "([{":
                    rb = _find_match(out, j, out[j], {"(": ")", "[": "]", "{": "}"}[out[j]])
                    if rb == -1:
                        i += 1
                        continue
                    exp_str = out[j : rb + 1]
                    base_start, base_tok = _token_left(out, i)
                    if base_tok == "":
                        i = rb + 1
                        continue
                    repl = f"({base_tok})**({exp_str})"
                    out = out[:base_start] + repl + out[rb + 1 :]
                    i = base_start + len(repl)
                    continue
                j = i + 1
                while j < len(out) and re.match(r"[A-Za-z0-9_.]", out[j]):
                    j += 1
                exp_str = out[i + 1 : j]
                base_start, base_tok = _token_left(out, i)
                if base_tok == "":
                    i = j
                    continue
                repl = f"({base_tok})**({exp_str})"
                out = out[:base_start] + repl + out[j:]
                i = base_start + len(repl)
                continue
        i += 1
    return out


def _cleanup_latex_misc(s: str) -> str:
    out = s.replace(r"\left", "").replace(r"\right", "").replace(r"\cdot", "*")
    out = re.sub(r"\\(,|;|:|!|quad|qquad)\b", " ", out)
    out = re.sub(r"\|(.*?)\|", r"abs(\1)", out)
    return out


# ---------- Formula conversion ----------
def normalize_formula(formula: str) -> Tuple[str, int]:
    formula = formula.strip()
    left = formula.split("=", 1)[0] if "=" in formula else formula
    m = re.search(r"u\[n([+-]?\d*)\]", left)
    base_offset = int(m.group(1) or 0) if m else 0

    def _norm_u(match):
        off = int(match.group(1) or "0")
        norm = off - base_offset
        return "u[n]" if norm == 0 else f"u[n{norm:+}]"

    normalized_u = re.sub(r"u\[n([+-]?\d*)\]", _norm_u, formula)
    if base_offset == 0:
        return normalized_u, base_offset
    normalized_n = re.sub(r"\bn\b", f"(n{ -base_offset:+})", normalized_u)
    return normalized_n, base_offset


def convert_formula_to_python(formula: str) -> str:
    rhs = formula.split("=", 1)[1] if "=" in formula else formula
    out = _cleanup_latex_misc(rhs)
    out = _replace_all_frac(out)
    out = _replace_all_sqrt(out)
    out = _replace_all_power(out)
    out = re.sub(r"cotg\((.*?)\)", r"(1/tan(\1))", out)
    out = out.replace("^", "**").replace("ln(", "log(")

    out = re.sub(r"\\log_\{([^{}]+)\}\{([^{}]+)\}", lambda m: f"log({m.group(2)},{m.group(1)})", out)
    out = re.sub(r"\\log\[(.*?)\]\{(.*?)\}", lambda m: f"log({m.group(2)},{m.group(1)})", out)
    out = out.replace("\\log", "log").replace("\\ln", "log")

    trig_map = {
        r"\\sin": "sin",
        r"\\cos": "cos",
        r"\\tan": "tan",
        r"\\cot": "1/tan",
        r"\\sec": "1/cos",
        r"\\csc": "1/sin",
        r"\\sinh": "sinh",
        r"\\cosh": "cosh",
        r"\\tanh": "tanh",
    }
    for latex_cmd, py_func in trig_map.items():
        out = re.sub(latex_cmd + r"\{([^{}]+)\}", f"{py_func}(\\1)", out)
        out = re.sub(latex_cmd + r"\s*([A-Za-z0-9_\(])", f"{py_func}(\\1", out)

    return re.sub(r"\s+", " ", out).strip()


# ---------- Compute ----------
def compute_sequence(k: int, initials: Dict[int, float], n_target: int, python_rhs: str) -> Dict[int, float]:
    values = dict(initials)
    offsets = sorted({int(m or 0) for m in re.findall(r"u\[n([+-]?\d*)\]", python_rhs)})
    if not offsets:
        raise ValueError("Không tìm thấy các u[n+...] trong công thức.")

    max_offset = max(offsets)
    start_n = max(values.keys()) + 1 if values else k + max_offset + 1

    for n in range(start_n, n_target + 1):
        def repl(m):
            off = int(m.group(1) or "0")
            idx = n + off
            if idx not in values:
                raise KeyError(f"Thiếu giá trị u({idx}) khi tính u({n}).")
            return repr(values[idx])

        expr_filled = re.sub(r"u\[n([+-]?\d*)\]", repl, python_rhs)
        try:
            values[n] = float(safe_eval(expr_filled, names={"n": n}))
        except Exception as e:
            raise RuntimeError(f"Lỗi tính u({n}): {e}")
    return values


# ---------- UI ----------
st.title("Tính dãy số theo công thức truy hồi")
st.markdown(
    "Nhập công thức (hỗ trợ LaTeX: `\\frac`, `\\dfrac`, `\\sqrt[k]{}`, mũ `^{}`, `|.|`, `\\cdot`, `\\log_{a}{b}`.)"
)

with st.sidebar:
    st.header("Phím tắt")

    def _ins(s: str):
        st.session_state.setdefault("raw_formula", "")
        if st.session_state["raw_formula"] and not st.session_state["raw_formula"].endswith((" ", "\n")):
            st.session_state["raw_formula"] += " "
        st.session_state["raw_formula"] += s + " "

    row = st.columns(4)
    if row[0].button("sin", use_container_width=True):
        _ins(r"\sin{}")
    if row[1].button("cos", use_container_width=True):
        _ins(r"\cos{}")
    if row[2].button("tan", use_container_width=True):
        _ins(r"\tan{}")
    if row[3].button("cot", use_container_width=True):
        _ins(r"\cot{}")

    row = st.columns(4)
    if row[0].button("Phân số", use_container_width=True):
        _ins(r"\frac{}{}")
    if row[1].button("Lũy thừa", use_container_width=True):
        _ins(r"^{}")
    if row[2].button("√", use_container_width=True):
        _ins(r"\sqrt{}")
    if row[3].button("√k", use_container_width=True):
        _ins(r"\sqrt[]{}")

raw_formula = st.text_area(
    "Công thức",
    value=st.session_state.get("raw_formula", "u[n]=u[n-1]+2*u[n-2]"),
    height=140,
)
st.session_state["raw_formula"] = raw_formula

c1, c2, c3 = st.columns(3)
with c1:
    k = st.number_input("Chỉ số bắt đầu", value=0, format="%d")
with c2:
    target_n = st.number_input("Tính tới chỉ số", value=5, format="%d")
with c3:
    display_count = st.number_input("Số hạng hiển thị (gần nhất)", value=10, min_value=1, format="%d")

st.markdown("**Xem trước**")
try:
    norm, base = normalize_formula(raw_formula)

    def _latex_preview(expr: str) -> str:
        s = re.sub(r"u\[([^\]]+)\]", r"u_{\1}", expr)
        s = re.sub(r"(?<!\\)\bsqrt\s*\((.*?)\)", r"\\sqrt{\1}", s)
        s = s.replace("*", r" \cdot ")
        return re.sub(r"\s+", " ", s).strip()

    st.latex(_latex_preview(norm))
except Exception as e:
    st.error(f"Không thể preview: {e}")

if st.button("Chuyển đổi"):
    try:
        normalized, base_offset = normalize_formula(raw_formula)
        python_rhs = convert_formula_to_python(normalized)
        st.session_state.update(
            {
                "normalized": normalized,
                "python_rhs": python_rhs,
                "base_offset": base_offset,
            }
        )
        st.code(python_rhs, language="python")

        rhs = normalized.split("=", 1)[1] if "=" in normalized else normalized
        offsets = sorted({int(m or 0) for m in re.findall(r"u\[n([+-]?\d*)\]", rhs)})
        needed_indices = [k + off for off in offsets]
        st.session_state["needed_indices"] = needed_indices
        st.info(f"Cần giá trị ban đầu cho các chỉ số: {sorted(set(needed_indices))}")
    except Exception as e:
        st.error(f"Phân tích thất bại: {e}")

needed = st.session_state.get("needed_indices", [])
if needed:
    st.markdown("### Giá trị khởi đầu")
    idxs = sorted(set(needed))
    prev_vals = [st.session_state.get(f"init_{i}", "") for i in idxs]
    df_init = pd.DataFrame({"index": idxs, "u(index)": prev_vals})
    edited_df = st.data_editor(df_init, key="init_editor", num_rows="fixed", use_container_width=True)

    if st.button("Tính"):
        try:
            initials = {}
            for _, row in edited_df.iterrows():
                if pd.isna(row.get("index", None)):
                    continue
                i = int(row["index"])
                v = str(row.get("u(index)", "")).strip()
                if v != "":
                    initials[i] = float(v)

            missing = [i for i in idxs if i not in initials]
            if missing:
                st.error(f"Thiếu giá trị khởi đầu cho: {missing}")
                st.stop()

            normalized = st.session_state["normalized"]
            python_rhs = st.session_state["python_rhs"]
            values = compute_sequence(k, initials, int(target_n), python_rhs)

            st.success("Hoàn tất")

            rows = [{"Chỉ số": i, "Giá trị": values[i]} for i in sorted(values.keys())]
            df = pd.DataFrame(rows)
            if len(df) > display_count:
                df = df.tail(display_count)

            st.markdown(
                """
                <style>
                table { width: 100% !important; }
                th, td { text-align: center !important; padding: 8px !important; }
                </style>
                """,
                unsafe_allow_html=True,
            )
            st.write(df.style.hide(axis="index").to_html(), unsafe_allow_html=True)

            st.markdown("### Biểu đồ")
            n_list = sorted(values.keys())
            df_seq = pd.DataFrame({"n": n_list, "u(n)": [values[i] for i in n_list]}).set_index("n")
            st.subheader(r"u(n)")
            st.line_chart(df_seq, height=280, use_container_width=True)

            st.subheader(r"$\Delta(u(n)) = |u(n+1)-u(n)|$ + tham chiếu")
            n_diff = [n for n in n_list if (n + 1) in values]
            diffs_u = [abs(values[n + 1] - values[n]) for n in n_diff]

            ref_harm, ref_sq = [], []
            for n in n_diff:
                if n in (-1, 0):
                    ref_harm.append(float("nan"))
                    ref_sq.append(float("nan"))
                else:
                    ref_harm.append(abs(1 / (n + 1) - 1 / n))
                    ref_sq.append(abs(1 / ((n + 1) ** 2) - 1 / (n**2)))

            df_diff = pd.DataFrame(
                {
                    "n": n_diff,
                    "|u(n+1)-u(n)|": diffs_u,
                    "|1/(n+1)-1/n|": ref_harm,
                    "|1/(n+1)^2-1/n^2|": ref_sq,
                }
            ).set_index("n")
            st.line_chart(df_diff, height=320, use_container_width=True)
            st.caption("Bỏ qua n = -1, 0 ở các đường tham chiếu để tránh chia 0.")
        except Exception as e:
            st.error(f"Lỗi khi tính: {e}")
