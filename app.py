j"""
Smart Oil Consumption Monitoring System — 100% Python build
UI: Streamlit | DB: SQLite | ML: Scikit-learn (+ saved models via joblib)
Charts: Plotly | Reports: fpdf2

Run locally:    streamlit run app.py
Deploy free:    push this folder to GitHub, deploy at https://share.streamlit.io
"""

import os
import sqlite3
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from fpdf import FPDF
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# ---------------- CONFIG ----------------
DB_FILE = "oil_monitor.db"
MODEL_DIR = "models"
CATEGORIES = ["Household", "Vehicle", "Industrial"]
DEMO_USER = "demo_user"

st.set_page_config(page_title="Smart Oil Consumption Monitor", page_icon="🛢️", layout="wide")
os.makedirs(MODEL_DIR, exist_ok=True)


# ---------------- DATABASE LAYER (SQLite) ----------------
def get_conn():
    return sqlite3.connect(DB_FILE, check_same_thread=False)


def init_db():
    conn = get_conn()
    conn.execute(
        """CREATE TABLE IF NOT EXISTS usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT NOT NULL,
            date TEXT NOT NULL,
            category TEXT NOT NULL,
            liters REAL NOT NULL,
            cost REAL NOT NULL
        )"""
    )
    conn.commit()
    conn.close()


def insert_entry(user, date, category, liters, cost):
    conn = get_conn()
    conn.execute(
        "INSERT INTO usage (user, date, category, liters, cost) VALUES (?, ?, ?, ?, ?)",
        (user, str(date), category, liters, cost),
    )
    conn.commit()
    conn.close()


def load_data(user):
    conn = get_conn()
    df = pd.read_sql(
        "SELECT * FROM usage WHERE user = ? ORDER BY date", conn, params=(user,), parse_dates=["date"]
    )
    conn.close()
    return df


def seed_demo_data():
    """Populate demo_user with 90 days of realistic simulated data, once only."""
    conn = get_conn()
    count = conn.execute("SELECT COUNT(*) FROM usage WHERE user = ?", (DEMO_USER,)).fetchone()[0]
    conn.close()
    if count > 0:
        return

    rng = np.random.default_rng(42)
    start = datetime.today() - timedelta(days=90)
    rows = []
    for cat, base in [("Household", 3.0), ("Vehicle", 8.0), ("Industrial", 25.0)]:
        for i in range(90):
            date = (start + timedelta(days=i)).strftime("%Y-%m-%d")
            seasonal = 1 + 0.15 * np.sin(i / 12)
            noise = rng.normal(0, 0.08)
            liters = max(0.2, base * seasonal * (1 + noise))
            cost = round(liters * 1.35, 2)
            rows.append((DEMO_USER, date, cat, round(liters, 2), cost))

    conn = get_conn()
    conn.executemany("INSERT INTO usage (user, date, category, liters, cost) VALUES (?, ?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()


# ---------------- ML LAYER (Scikit-learn + joblib) ----------------
def build_features(df):
    df = df.copy()
    df["day_index"] = (df["date"] - df["date"].min()).dt.days
    df["sin_day"] = np.sin(df["day_index"] / 12)
    df["cos_day"] = np.cos(df["day_index"] / 12)
    return df


def model_path(user, category):
    safe_user = "".join(c for c in user if c.isalnum()) or "user"
    return os.path.join(MODEL_DIR, f"model_{safe_user}_{category}.joblib")


def train_and_save_model(df, user, category):
    """Trains a LinearRegression model on day/seasonal features, evaluates on a held-out
    split, saves the model with joblib, and returns evaluation metrics."""
    sub = df[df["category"] == category].sort_values("date")
    if len(sub) < 10:
        return None

    sub = build_features(sub)
    X = sub[["day_index", "sin_day", "cos_day"]].values
    y = sub["liters"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    model = LinearRegression().fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {"mae": mean_absolute_error(y_test, preds), "r2": r2_score(y_test, preds), "n_samples": len(sub)}

    # Refit on all data for the best possible forecast, then persist
    final_model = LinearRegression().fit(X, y)
    joblib.dump(final_model, model_path(user, category))
    return metrics


def load_model(user, category):
    path = model_path(user, category)
    if os.path.exists(path):
        return joblib.load(path)
    return None


def forecast_next_period(df, user, category, periods=7):
    model = load_model(user, category)
    sub = df[df["category"] == category].sort_values("date")
    if model is None or sub.empty:
        return None, None

    sub = build_features(sub)
    last_day = sub["day_index"].max()
    future_idx = np.arange(last_day + 1, last_day + 1 + periods)
    future_X = np.column_stack([future_idx, np.sin(future_idx / 12), np.cos(future_idx / 12)])
    preds = np.clip(model.predict(future_X), 0, None)

    future_dates = [sub["date"].max() + timedelta(days=int(d)) for d in range(1, periods + 1)]
    forecast_df = pd.DataFrame({"date": future_dates, "predicted_liters": preds})
    trend_per_day = model.coef_[0]
    return forecast_df, trend_per_day


def detect_spike(df, category, threshold_pct=25):
    sub = df[df["category"] == category].sort_values("date")
    if len(sub) < 6:
        return False, 0
    rolling_avg = sub["liters"].iloc[-6:-1].mean()
    latest = sub["liters"].iloc[-1]
    if rolling_avg == 0:
        return False, 0
    pct_change = ((latest - rolling_avg) / rolling_avg) * 100
    return pct_change >= threshold_pct, pct_change


def recommendation_text(trend_per_day, spike_detected):
    if spike_detected:
        return "⚠️ Usage spiked well above your recent average — check for leaks, idle equipment, or unusual activity."
    if trend_per_day is None:
        return "Train the model (Predictions tab) to unlock personalized recommendations."
    if trend_per_day > 0.05:
        return "📈 Usage is trending upward. Consider auditing equipment efficiency or setting a reduction goal."
    if trend_per_day < -0.05:
        return "📉 Usage is trending down. Keep up current efficiency habits."
    return "➡️ Usage is stable. No action needed right now."


# ---------------- REPORT LAYER (fpdf2) ----------------
def generate_pdf_report(user, df, summary_rows):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Smart Oil Consumption Monitoring Report", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"User: {user}", ln=True)
    pdf.cell(0, 8, f"Generated: {datetime.today().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Summary by Category", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for row in summary_rows:
        pdf.cell(0, 7, row, ln=True)

    if not df.empty:
        chart_path = "temp_chart.png"
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 3))
        for cat in CATEGORIES:
            sub = df[df["category"] == cat].sort_values("date")
            if not sub.empty:
                ax.plot(sub["date"], sub["liters"], label=cat)
        ax.set_ylabel("Liters")
        ax.legend()
        fig.tight_layout()
        fig.savefig(chart_path, dpi=120)
        plt.close(fig)

        pdf.ln(4)
        pdf.image(chart_path, w=180)
        os.remove(chart_path)

    return bytes(pdf.output())


# ---------------- APP ----------------
init_db()
seed_demo_data()

st.title("🛢️ Smart Oil Consumption Monitoring System")
st.caption("Built entirely in Python — Streamlit, SQLite, Scikit-learn, Plotly, and fpdf2.")

with st.sidebar:
    st.header("👤 User")
    username = st.text_input("Username", value=DEMO_USER, help=f"Try '{DEMO_USER}' to see sample data.")

    st.divider()
    st.header("Log New Usage")
    entry_date = st.date_input("Date", value=datetime.today())
    entry_category = st.selectbox("Category", CATEGORIES)
    entry_liters = st.number_input("Liters used", min_value=0.0, step=0.1)
    entry_cost = st.number_input("Cost ($)", min_value=0.0, step=0.5)
    if st.button("Add Entry", use_container_width=True):
        insert_entry(username, entry_date, entry_category, entry_liters, entry_cost)
        st.success("Entry logged!")
        st.rerun()

    st.divider()
    goal = st.slider("🎯 Reduction goal (%)", 0, 50, 10)
    st.caption(f"Target: reduce usage by {goal}% vs. current trend.")

data = load_data(username)

tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔮 Predictions", "🔔 Alerts", "📄 Report"])

# ---- Dashboard ----
with tab1:
    cols = st.columns(3)
    for col, cat in zip(cols, CATEGORIES):
        total = data[data["category"] == cat]["liters"].sum() if not data.empty else 0
        col.metric(f"{cat} — Total Liters", f"{total:.1f} L")

    st.subheader("Consumption Over Time")
    if not data.empty:
        fig = px.line(data, x="date", y="liters", color="category", markers=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data yet. Log an entry from the sidebar, or use the demo_user username.")

    st.subheader("Raw Data")
    st.dataframe(data.sort_values("date", ascending=False), use_container_width=True)

# ---- Predictions ----
with tab2:
    st.subheader("Train & Evaluate Model")
    pred_category = st.selectbox("Category", CATEGORIES, key="pred_cat")

    if st.button("Train / Update Model"):
        metrics = train_and_save_model(data, username, pred_category)
        if metrics:
            st.session_state["last_metrics"] = metrics
            st.success("Model trained and saved.")
        else:
            st.warning("Need at least 10 logged entries for this category to train a model.")

    if "last_metrics" in st.session_state:
        m = st.session_state["last_metrics"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean Absolute Error", f"{m['mae']:.3f} L")
        c2.metric("R² Score", f"{m['r2']:.3f}")
        c3.metric("Samples used", m["n_samples"])

    st.divider()
    st.subheader("Forecast")
    periods = st.slider("Days to forecast", 3, 30, 7)
    forecast_df, trend = forecast_next_period(data, username, pred_category, periods)
    if forecast_df is not None:
        fig = px.line(forecast_df, x="date", y="predicted_liters", markers=True)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(forecast_df, use_container_width=True)
    else:
        st.info("Train a model above first to generate a forecast.")

# ---- Alerts ----
with tab3:
    st.subheader("Alerts & Personalized Recommendations")
    for cat in CATEGORIES:
        spike, pct = detect_spike(data, cat)
        _, trend = forecast_next_period(data, username, cat, periods=1)
        st.markdown(f"**{cat}**")
        if spike:
            st.error(f"Spike detected: latest usage is {pct:.1f}% above recent average.")
        else:
            st.success("No abnormal spikes detected.")
        st.write(recommendation_text(trend, spike))
        st.divider()

# ---- Report ----
with tab4:
    st.subheader("Download a PDF Summary")
    summary_rows = [
        f"{cat}: {data[data['category'] == cat]['liters'].sum():.1f} L total"
        for cat in CATEGORIES
    ]
    if st.button("Generate PDF Report"):
        pdf_bytes = generate_pdf_report(username, data, summary_rows)
        st.download_button(
            "⬇️ Download Report",
            data=pdf_bytes,
            file_name=f"oil_report_{username}.pdf",
            mime="application/pdf",
        )
