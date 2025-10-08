# Sequence Calculator — Recursive Formula Evaluator
A **Streamlit web app** that lets you define and compute **recursive sequences** using familiar **LaTeX-style formulas**.  
It automatically parses your formula, evaluates the sequence safely, and visualizes the results — all in your browser.

---

## Live Demo

[**Open in Streamlit Cloud**](https://computing-sequence-npbk-apcs.streamlit.app/) 

---

What it does

- Accepts recursive formulas written in LaTeX syntax  
  *(e.g. `\frac`, `\sqrt[k]{}`, `^{}`, `|.|`, `\log_{a}{b}`, `\sin`, `\cos`, `\tan`, ...)*  
- Converts the LaTeX expression into **safe executable Python code**  
- Lets you **input initial values** interactively  
- Generates **tables and charts** of the computed sequence  
- Compares with reference sequences like \( \frac{1}{n} \) or \( \frac{1}{n^2} \)  
- Fully runs in the browser via Streamlit — no local setup hassle
