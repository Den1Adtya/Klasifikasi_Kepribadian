import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, roc_auc_score
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Personality Classifier – Random Forest",
    page_icon="🌲",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    .metric-card {
        background: #f0f4ff;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        margin-bottom: 0.5rem;
        border-left: 4px solid #4f6df5;
    }
    h1 { color: #1a1a2e; }
    h2, h3 { color: #2c3e7a; }
    .stDataFrame { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Helper: load & preprocess data ───────────────────────────────────────────
@st.cache_data
def load_and_preprocess(uploaded_file):
    df = pd.read_csv(uploaded_file)

    # Fix possible typo in column names (strip spaces)
    df.columns = df.columns.str.strip()

    info_lines = []
    buf = io.StringIO()
    df.info(buf=buf)
    info_lines = buf.getvalue()

    return df, info_lines


def preprocess(df):
    df = df.copy()
    df.drop_duplicates(inplace=True)

    # Encode binary columns
    df["Stage_fear"] = df["Stage_fear"].replace({"Yes": 1, "No": 0})
    df["Drained_after_socializing"] = df["Drained_after_socializing"].replace({"Yes": 1, "No": 0})
    df["Personality"] = df["Personality"].replace({"Extrovert": 1, "Introvert": 0})

    X = df.drop("Personality", axis=1)
    y = df["Personality"]

    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)

    return X_scaled, y, sc, df, X.columns.tolist()


@st.cache_resource
def train_model(X_train, y_train, n_estimators, max_depth, min_samples_split):
    rfc = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth if max_depth > 0 else None,
        min_samples_split=min_samples_split,
        random_state=42,
    )
    rfc.fit(X_train, y_train)
    return rfc


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/forest.png", width=80)
    st.title("⚙️ Konfigurasi")

    st.subheader("Upload Dataset")
    uploaded = st.file_uploader("Upload file CSV", type=["csv"])

    st.subheader("Hyperparameter")
    n_estimators     = st.slider("Jumlah Pohon (n_estimators)", 50, 500, 100, step=50)
    max_depth        = st.slider("Max Depth (0 = None)", 0, 30, 0)
    min_samples_split= st.slider("Min Samples Split", 2, 20, 2)
    test_size        = st.slider("Test Size (%)", 10, 40, 25)

    run_btn = st.button("🚀 Jalankan Model", use_container_width=True)

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("🌲 Personality Classifier – Random Forest")
st.markdown("Klasifikasi kepribadian **Introvert vs Extrovert** menggunakan Ensemble Learning – Random Forest.")

if uploaded is None:
    st.info("👈 Silakan upload file **personality_dataset.csv** di sidebar untuk memulai.")
    st.stop()

# Load raw data
df_raw, df_info = load_and_preprocess(uploaded)

# ── Tab layout ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Data Overview",
    "📊 Visualisasi EDA",
    "🤖 Hasil Model",
    "📈 Evaluasi",
    "🔮 Prediksi Baru",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 – Data Overview
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.header("📋 Data Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Jumlah Baris", df_raw.shape[0])
    col2.metric("Jumlah Kolom", df_raw.shape[1])
    col3.metric("Duplikasi", int(df_raw.duplicated().sum()))

    st.subheader("10 Data Terakhir")
    st.dataframe(df_raw.tail(10), use_container_width=True)

    st.subheader("Statistik Deskriptif")
    st.dataframe(df_raw.describe(), use_container_width=True)

    st.subheader("Info Tipe Data")
    st.code(df_info, language="text")

    st.subheader("Missing Values")
    missing = df_raw.isnull().sum().reset_index()
    missing.columns = ["Kolom", "Jumlah Missing"]
    st.dataframe(missing, use_container_width=True)

    dup_pct = (df_raw.duplicated().sum() / df_raw.shape[0]) * 100
    st.info(f"Persentase duplikasi: **{dup_pct:.2f}%**")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 – EDA
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.header("📊 Exploratory Data Analysis")

    # Scatter plot
    st.subheader("Time Spent Alone vs Social Event Attendance")
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    for label, color in [("Extrovert", "#4f6df5"), ("Introvert", "#e74c3c")]:
        sub = df_raw[df_raw["Personality"] == label]
        ax1.scatter(sub["Time_spent_Alone"], sub["Social_event_attendance"],
                    label=label, alpha=0.6, color=color)
    ax1.set_xlabel("Time Spent Alone")
    ax1.set_ylabel("Social Event Attendance")
    ax1.set_title("Personality: Time Spent Alone vs Social Event Attendance")
    ax1.legend()
    st.pyplot(fig1)

    # Bar plot
    st.subheader("Friends Circle Size vs Post Frequency")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    rounded_x = df_raw["Friends_circle_size"].round(0)
    sns.barplot(x=rounded_x, y=df_raw["Post_frequency"], ax=ax2, palette="Blues_d")
    ax2.set_xlabel("Friends Circle Size")
    ax2.set_ylabel("Post Frequency")
    ax2.set_title("Friends Circle Size vs Post Frequency")
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    # Distribution target
    st.subheader("Distribusi Kelas Target")
    fig3, ax3 = plt.subplots(figsize=(5, 4))
    counts = df_raw["Personality"].value_counts()
    ax3.pie(counts, labels=counts.index, autopct="%1.1f%%",
            colors=["#4f6df5", "#e74c3c"], startangle=90)
    ax3.set_title("Distribusi Introvert vs Extrovert")
    st.pyplot(fig3)

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    df_enc = df_raw.copy()
    df_enc["Stage_fear"] = df_enc["Stage_fear"].replace({"Yes": 1, "No": 0})
    df_enc["Drained_after_socializing"] = df_enc["Drained_after_socializing"].replace({"Yes": 1, "No": 0})
    df_enc["Personality"] = df_enc["Personality"].replace({"Extrovert": 1, "Introvert": 0})
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    sns.heatmap(df_enc.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax4)
    ax4.set_title("Correlation Heatmap")
    st.pyplot(fig4)

# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing & Training (shared across tabs 3-5)
# ─────────────────────────────────────────────────────────────────────────────
X_scaled, y, sc, df_clean, feature_names = preprocess(df_raw)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=test_size / 100, random_state=3
)

if run_btn or True:   # auto-train on upload
    rf = train_model(X_train, y_train, n_estimators, max_depth, min_samples_split)

    y_pred   = rf.predict(X_test)
    y_pred_t = rf.predict(X_train)
    y_scores = rf.predict_proba(X_test)[:, 1]

    train_acc = accuracy_score(y_train, y_pred_t)
    test_acc  = accuracy_score(y_test,  y_pred)

    cv_scores = cross_val_score(rf, X_scaled, y, cv=5, scoring="accuracy")

    fpr, tpr, _ = roc_curve(y_test, y_scores)
    auc          = roc_auc_score(y_test, y_scores)
    cm           = confusion_matrix(y_test, y_pred)
    report       = classification_report(y_test, y_pred,
                                         target_names=["Introvert", "Extrovert"],
                                         output_dict=True)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 3 – Hasil Model
    # ─────────────────────────────────────────────────────────────────────────
    with tab3:
        st.header("🤖 Hasil Model Random Forest")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Train Accuracy", f"{train_acc:.4f}")
        col2.metric("Test Accuracy",  f"{test_acc:.4f}")
        col3.metric("CV Mean (5-fold)", f"{cv_scores.mean():.4f}")
        col4.metric("AUC-ROC", f"{auc:.4f}")

        st.subheader("Cross-Validation Scores (5-Fold)")
        cv_df = pd.DataFrame({
            "Fold": [f"Fold {i+1}" for i in range(5)],
            "Accuracy": cv_scores
        })
        fig_cv, ax_cv = plt.subplots(figsize=(7, 3))
        ax_cv.bar(cv_df["Fold"], cv_df["Accuracy"], color="#4f6df5")
        ax_cv.axhline(cv_scores.mean(), color="red", linestyle="--", label=f"Mean = {cv_scores.mean():.4f}")
        ax_cv.set_ylim(0.8, 1.01)
        ax_cv.set_ylabel("Accuracy")
        ax_cv.set_title("5-Fold Cross Validation")
        ax_cv.legend()
        st.pyplot(fig_cv)

        st.subheader("Feature Importance")
        importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=True)
        fig_fi, ax_fi = plt.subplots(figsize=(7, 5))
        importances.plot(kind="barh", ax=ax_fi, color="#4f6df5")
        ax_fi.set_title("Feature Importance – Random Forest")
        ax_fi.set_xlabel("Importance Score")
        st.pyplot(fig_fi)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 4 – Evaluasi
    # ─────────────────────────────────────────────────────────────────────────
    with tab4:
        st.header("📈 Evaluasi Model")

        # Classification report
        st.subheader("Classification Report")
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)

        col_cm, col_roc = st.columns(2)

        with col_cm:
            st.subheader("Confusion Matrix")
            fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                          display_labels=["Introvert", "Extrovert"])
            disp.plot(cmap="Blues", ax=ax_cm)
            ax_cm.set_title("Confusion Matrix – Random Forest")
            st.pyplot(fig_cm)

        with col_roc:
            st.subheader("ROC Curve")
            fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
            ax_roc.plot(fpr, tpr, color="blue", lw=2, label=f"AUC = {auc:.4f}")
            ax_roc.plot([0, 1], [0, 1], color="red", linestyle="--")
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.set_title("ROC Curve")
            ax_roc.legend()
            st.pyplot(fig_roc)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 5 – Prediksi Baru
    # ─────────────────────────────────────────────────────────────────────────
    with tab5:
        st.header("🔮 Prediksi Kepribadian Baru")
        st.markdown("Masukkan nilai fitur untuk memprediksi apakah seseorang **Introvert** atau **Extrovert**.")

        with st.form("prediction_form"):
            c1, c2 = st.columns(2)
            with c1:
                time_alone   = st.slider("Time Spent Alone (jam/hari)", 0, 11, 5)
                stage_fear   = st.selectbox("Stage Fear", ["No", "Yes"])
                social_event = st.slider("Social Event Attendance (kali/bulan)", 0, 10, 3)
                going_out    = st.slider("Going Outside (kali/minggu)", 0, 7, 3)
            with c2:
                drained       = st.selectbox("Drained After Socializing", ["No", "Yes"])
                friends_size  = st.slider("Friends Circle Size", 0, 20, 8)
                post_freq     = st.slider("Post Frequency (kali/bulan)", 0, 20, 5)

            submitted = st.form_submit_button("🔍 Prediksi", use_container_width=True)

        if submitted:
            input_data = np.array([[
                time_alone,
                1 if stage_fear == "Yes" else 0,
                social_event,
                going_out,
                1 if drained == "Yes" else 0,
                friends_size,
                post_freq,
            ]])
            input_scaled = sc.transform(input_data)
            prediction   = rf.predict(input_scaled)[0]
            proba        = rf.predict_proba(input_scaled)[0]

            label = "Extrovert 🎉" if prediction == 1 else "Introvert 🤫"
            color = "#4f6df5" if prediction == 1 else "#e74c3c"

            st.markdown(f"""
            <div style="background:{color}22; border-left:5px solid {color};
                        border-radius:8px; padding:1.2rem 1.5rem; margin-top:1rem;">
                <h2 style="color:{color}; margin:0;">Hasil Prediksi: {label}</h2>
                <p style="margin:0.5rem 0 0;">
                    Probabilitas Introvert: <b>{proba[0]*100:.1f}%</b> &nbsp;|&nbsp;
                    Probabilitas Extrovert: <b>{proba[1]*100:.1f}%</b>
                </p>
            </div>
            """, unsafe_allow_html=True)

            fig_prob, ax_prob = plt.subplots(figsize=(5, 2.5))
            bars = ax_prob.barh(["Introvert", "Extrovert"], proba,
                                 color=["#e74c3c", "#4f6df5"])
            ax_prob.set_xlim(0, 1)
            ax_prob.set_xlabel("Probabilitas")
            ax_prob.set_title("Probabilitas Prediksi")
            for bar, val in zip(bars, proba):
                ax_prob.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                             f"{val*100:.1f}%", va="center")
            st.pyplot(fig_prob)
