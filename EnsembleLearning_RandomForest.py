"""
EnsembleLearning_RandomForest.py
---------------------------------
Script lengkap untuk klasifikasi Introvert/Extrovert
menggunakan Random Forest Classifier.

Jalankan:
    python EnsembleLearning_RandomForest.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
)

# ─── 1. Load Data ─────────────────────────────────────────────────────────────
print("=" * 60)
print("1. LOAD DATA")
print("=" * 60)

df = pd.read_csv("personality_datasert.csv")  # pastikan nama file benar
print(df.tail(10))

# ─── 2. Identifikasi Data ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. IDENTIFIKASI DATA")
print("=" * 60)

print("\n--- df.info() ---")
df.info()

print("\n--- Duplikasi ---")
percentage_duplicates = (df.duplicated().sum() / df.shape[0]) * 100
print(f"Jumlah duplikasi  : {df.duplicated().sum()}")
print(f"Persentase duplikasi: {percentage_duplicates:.2f}%")

print("\n--- df.describe() ---")
print(df.describe())

# ─── 3. Cleaning Data ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. CLEANING DATA")
print("=" * 60)

df.drop_duplicates(inplace=True)
print(f"Jumlah duplikasi setelah dihapus: {df.duplicated().sum()}")
print(f"Ukuran data setelah cleaning     : {df.shape}")

# ─── 4. Visualisasi EDA ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. VISUALISASI EDA")
print("=" * 60)

# Scatter plot
plt.figure(figsize=(10, 6))
for label, color in [("Extrovert", "steelblue"), ("Introvert", "tomato")]:
    sub = df[df["Personality"] == label]
    plt.scatter(sub["Time_spent_Alone"], sub["Social_event_attendance"],
                label=label, alpha=0.6, color=color)
plt.title("Personality: Time Spent Alone vs Social Event Attendance")
plt.xlabel("Time Spent Alone")
plt.ylabel("Social Event Attendance")
plt.legend()
plt.tight_layout()
plt.savefig("plot_scatter.png", dpi=150)
plt.show()
print("Scatter plot disimpan: plot_scatter.png")

# Bar plot
plt.figure(figsize=(13, 6))
sns.barplot(x=df["Friends_circle_size"].round(0), y=df["Post_frequency"], palette="Blues_d")
plt.xlabel("Friends Circle Size")
plt.ylabel("Post Frequency")
plt.title("Friends Circle Size vs Post Frequency")
plt.tight_layout()
plt.savefig("plot_barplot.png", dpi=150)
plt.show()
print("Bar plot disimpan: plot_barplot.png")

# ─── 5. Feature Engineering ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. FEATURE ENGINEERING")
print("=" * 60)

df["Stage_fear"] = df["Stage_fear"].replace({"Yes": 1, "No": 0})
df["Drained_after_socializing"] = df["Drained_after_socializing"].replace({"Yes": 1, "No": 0})
df["Personality"] = df["Personality"].replace({"Extrovert": 1, "Introvert": 0})

print(df.head())

# Correlation heatmap
plt.figure(figsize=(9, 7))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("plot_heatmap.png", dpi=150)
plt.show()
print("Heatmap disimpan: plot_heatmap.png")

# ─── 6. Persiapan Data ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. PERSIAPAN DATA UNTUK MODEL")
print("=" * 60)

sc = StandardScaler()
X = df.drop("Personality", axis=1)
y = df["Personality"]
feature_names = X.columns.tolist()
X_scaled = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=3
)
print(f"Training samples : {X_train.shape[0]}")
print(f"Testing  samples : {X_test.shape[0]}")

# ─── 7. Training Model ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. TRAINING MODEL – RANDOM FOREST")
print("=" * 60)

rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)

# ─── 8. Evaluasi Akurasi ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("8. EVALUASI AKURASI")
print("=" * 60)

yhat   = rfc.predict(X_test)
yhat_t = rfc.predict(X_train)

train_accuracy = accuracy_score(y_train, yhat_t)
test_accuracy  = accuracy_score(y_test, yhat)

Accuracy_score = pd.DataFrame({
    "Model": ["RandomForestClassifier"],
    "Train_Accuracy": [train_accuracy],
    "Test_Accuracy":  [test_accuracy],
})
print(Accuracy_score)

# ─── 9. Cross-Validation ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("9. CROSS-VALIDATION (5-FOLD)")
print("=" * 60)

cv_scores = cross_val_score(rfc, X_scaled, y, cv=5, scoring="accuracy")
print(f"CV Scores  : {cv_scores}")
print(f"CV Mean    : {cv_scores.mean():.4f}")
print(f"CV Std Dev : {cv_scores.std():.4f}")

plt.figure(figsize=(7, 4))
plt.bar([f"Fold {i+1}" for i in range(5)], cv_scores, color="steelblue")
plt.axhline(cv_scores.mean(), color="red", linestyle="--",
            label=f"Mean = {cv_scores.mean():.4f}")
plt.ylabel("Accuracy")
plt.title("5-Fold Cross Validation")
plt.legend()
plt.tight_layout()
plt.savefig("plot_cv.png", dpi=150)
plt.show()
print("CV plot disimpan: plot_cv.png")

# ─── 10. Feature Importance ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("10. FEATURE IMPORTANCE")
print("=" * 60)

importances = pd.Series(rfc.feature_importances_, index=feature_names).sort_values(ascending=True)
print(importances)

plt.figure(figsize=(8, 5))
importances.plot(kind="barh", color="steelblue")
plt.title("Feature Importance – Random Forest")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("plot_feature_importance.png", dpi=150)
plt.show()
print("Feature importance plot disimpan: plot_feature_importance.png")

# ─── 11. Model Terbaik & Prediksi ────────────────────────────────────────────
print("\n" + "=" * 60)
print("11. MODEL TERBAIK & PREDIKSI")
print("=" * 60)

best_model = rfc
y_pred = best_model.predict(X_test)
print(f"Akurasi pada data uji: {accuracy_score(y_test, y_pred):.4f}")

# Contoh prediksi 1 sample
sample_data = X_test[0].reshape(1, -1)
prediction = rfc.predict(sample_data)
print(f"\nContoh Prediksi (sample pertama dari X_test):")
print(f"  Prediction : {prediction[0]}")
print(f"  Label      : {'Extrovert' if prediction[0] == 1 else 'Introvert'}")

# ─── 12. Classification Report ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("12. LAPORAN KLASIFIKASI")
print("=" * 60)

print(classification_report(y_test, y_pred, target_names=["Introvert", "Extrovert"]))

# ─── 13. Confusion Matrix ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("13. CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(y_test, y_pred)
print(cm)

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(confusion_matrix=cm,
                       display_labels=["Introvert", "Extrovert"]).plot(cmap="Blues", ax=ax)
ax.set_title("Confusion Matrix – Random Forest")
plt.tight_layout()
plt.savefig("plot_confusion_matrix.png", dpi=150)
plt.show()
print("Confusion matrix disimpan: plot_confusion_matrix.png")

# ─── 14. ROC Curve ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("14. ROC CURVE & AUC")
print("=" * 60)

y_scores = best_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_scores)
auc = roc_auc_score(y_test, y_scores)

print(f"AUC-ROC: {auc:.4f}")

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {auc:.4f})")
plt.plot([0, 1], [0, 1], color="red", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Random Forest")
plt.legend()
plt.tight_layout()
plt.savefig("plot_roc_curve.png", dpi=150)
plt.show()
print("ROC curve disimpan: plot_roc_curve.png")

# ─── Kesimpulan ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("KESIMPULAN")
print("=" * 60)
print(f"""
Dataset   : Personality Dataset (Introvert vs Extrovert)
Fitur     : {', '.join(feature_names)}
Model     : Random Forest Classifier (n_estimators=100)

Hasil:
  - Train Accuracy      : {train_accuracy:.4f}
  - Test  Accuracy      : {test_accuracy:.4f}
  - CV Mean (5-fold)    : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}
  - AUC-ROC             : {auc:.4f}

Fitur paling penting  : {importances.idxmax()}
""")
