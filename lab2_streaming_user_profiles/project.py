import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score
)

warnings.filterwarnings("ignore")

DATA_PATH = "data/streaming_users.csv"
REPORTS_DIR = "reports"
FIGURES_DIR = "reports/figures"
MODELS_DIR = "models"


def create_project_structure():
    folders = [
        REPORTS_DIR,
        FIGURES_DIR,
        f"{FIGURES_DIR}/clustering",
        f"{FIGURES_DIR}/binary",
        f"{FIGURES_DIR}/multiclass",
        MODELS_DIR,
        "data"
    ]

    for folder in folders:
        os.makedirs(folder, exist_ok=True)

    print("✅ Структура проекта создана")


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Файл {DATA_PATH} не найден")

    df = pd.read_csv(DATA_PATH)
    print(f"✅ Загружено записей: {len(df):,}")
    print(f"✅ Количество признаков: {df.shape[1]}")

    return df


def prepare_features(df):
    feature_columns = [
        "age",
        "monthly_watch_hours",
        "avg_session_duration",
        "number_of_sessions",
        "movies_watched",
        "series_watched",
        "documentaries_watched",
        "subscription_months",
        "monthly_fee",
        "support_requests",
        "days_since_last_login",
        "likes_given",
        "ratings_count",
        "watchlist_items",
        "device_count"
    ]

    features_df = df[feature_columns].copy()
    features_df = features_df.fillna(features_df.median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_df)

    joblib.dump(scaler, f"{MODELS_DIR}/scaler.pkl")

    return features_df, X_scaled, feature_columns


def plot_correlation_matrix(features_df):
    plt.figure(figsize=(13, 10))
    corr = features_df.corr()
    sns.heatmap(corr, cmap="coolwarm", annot=False)
    plt.title("Корреляционная матрица признаков пользователей")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/binary/10_correlation_matrix.png", dpi=300)
    plt.close()


def perform_clustering(features_df, X_scaled):
    print("\n🔍 Кластеризация пользователей")

    silhouette_scores = {}
    for k in range(2, 8):
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        silhouette_scores[k] = score
        print(f"k={k}: Silhouette Score = {score:.3f}")

    optimal_k = max(silhouette_scores, key=silhouette_scores.get)
    print(f"✅ Оптимальное число кластеров: {optimal_k}")

    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
    cluster_labels = kmeans.fit_predict(X_scaled)

    joblib.dump(kmeans, f"{MODELS_DIR}/kmeans_model.pkl")

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap="tab10", s=25, alpha=0.7)
    plt.title("PCA-визуализация кластеров пользователей")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.colorbar(scatter, label="Кластер")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/clustering/01_pca_clusters.png", dpi=300)
    plt.close()

    sample_size = min(700, len(X_scaled))
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_scaled[:sample_size])

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cluster_labels[:sample_size], cmap="tab10", s=25, alpha=0.7)
    plt.title("t-SNE-визуализация кластеров пользователей")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.colorbar(scatter, label="Кластер")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/clustering/02_tsne_results.png", dpi=300)
    plt.close()

    dbscan = DBSCAN(eps=2.5, min_samples=10)
    dbscan_labels = dbscan.fit_predict(X_scaled)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels, cmap="tab20", s=25, alpha=0.7)
    plt.title("DBSCAN-анализ пользовательских профилей")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.colorbar(scatter, label="DBSCAN cluster")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/clustering/03_dbscan_analysis.png", dpi=300)
    plt.close()

    agglomerative = AgglomerativeClustering(n_clusters=optimal_k)
    aggl_labels = agglomerative.fit_predict(X_scaled)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=aggl_labels, cmap="tab10", s=25, alpha=0.7)
    plt.title("Agglomerative Clustering пользовательских профилей")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.colorbar(scatter, label="Кластер")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/clustering/04_agglomerative.png", dpi=300)
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()), marker="o")
    plt.title("Оценка качества кластеризации")
    plt.xlabel("Количество кластеров")
    plt.ylabel("Silhouette Score")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/clustering/05_clusters_validation.png", dpi=300)
    plt.close()

    cluster_profile = features_df.copy()
    cluster_profile["cluster"] = cluster_labels
    cluster_means = cluster_profile.groupby("cluster").mean()

    plt.figure(figsize=(12, 7))
    sns.heatmap(cluster_means, cmap="YlGnBu")
    plt.title("Средние значения признаков по кластерам")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/clustering/06_cluster_profiles.png", dpi=300)
    plt.close()

    return cluster_labels, optimal_k, silhouette_scores


def perform_binary_classification(features_df, cluster_labels, df):
    print("\n🎯 Бинарная классификация: прогноз оттока пользователя")

    X = features_df.copy()
    X["cluster_label"] = cluster_labels
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"Accuracy: {accuracy:.3f}")
    print(f"F1-macro: {f1:.3f}")
    print(f"ROC-AUC: {roc_auc:.3f}")

    joblib.dump(model, f"{MODELS_DIR}/binary_churn_classifier.pkl")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No churn", "Churn"], yticklabels=["No churn", "Churn"])
    plt.title("Матрица ошибок: прогноз оттока")
    plt.xlabel("Предсказано")
    plt.ylabel("Реально")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/binary/07_confusion_matrix.png", dpi=300)
    plt.close()

    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    plt.figure(figsize=(10, 7))
    sns.barplot(data=importance.head(12), x="importance", y="feature")
    plt.title("Важность признаков для прогноза оттока")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/binary/08_feature_importance.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.histplot(df["days_since_last_login"], bins=30)
    plt.title("Распределение дней с последнего входа")
    plt.xlabel("Дней с последнего входа")
    plt.ylabel("Количество пользователей")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/binary/09_last_login_distribution.png", dpi=300)
    plt.close()

    return {
        "model": model,
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
        "cv_f1": cv_scores.mean(),
        "feature_importance": importance
    }


def perform_multiclass_classification(features_df, cluster_labels, df):
    print("\n🎯 Многоклассовая классификация: тип пользователя")

    X = features_df.copy()
    X["cluster_label"] = cluster_labels

    le = LabelEncoder()
    y = le.fit_transform(df["user_type"])

    joblib.dump(le, f"{MODELS_DIR}/user_type_label_encoder.pkl")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=250, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1),
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7)
    }

    results = {}

    for name, model in models.items():
        print(f"\nМодель: {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_test, y_pred, average="macro", zero_division=0)

        print(f"Accuracy: {accuracy:.3f}")
        print(f"F1-macro: {f1_macro:.3f}")

        results[name] = {
            "model": model,
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "precision": precision,
            "recall": recall,
            "y_pred": y_pred
        }

    best_name = max(results.keys(), key=lambda name: results[name]["f1_macro"])
    best = results[best_name]

    joblib.dump(best["model"], f"{MODELS_DIR}/best_user_type_classifier.pkl")

    class_names = list(le.classes_)

    cm = confusion_matrix(y_test, best["y_pred"])
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Матрица ошибок: {best_name}")
    plt.xlabel("Предсказано")
    plt.ylabel("Реально")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/multiclass/11_confusion_matrix.png", dpi=300)
    plt.close()

    class_distribution = df["user_type"].value_counts()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=class_distribution.index, y=class_distribution.values)
    plt.title("Распределение типов пользователей")
    plt.xlabel("Тип пользователя")
    plt.ylabel("Количество")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/multiclass/12_class_distribution.png", dpi=300)
    plt.close()

    comparison_df = pd.DataFrame([
        {"model": name, "accuracy": r["accuracy"], "f1_macro": r["f1_macro"]}
        for name, r in results.items()
    ])

    plt.figure(figsize=(9, 6))
    comparison_df.set_index("model")[["accuracy", "f1_macro"]].plot(kind="bar")
    plt.title("Сравнение моделей многоклассовой классификации")
    plt.ylabel("Значение метрики")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/multiclass/13_models_comparison.png", dpi=300)
    plt.close()

    return {
        "best_model_name": best_name,
        "best": best,
        "results": results,
        "class_names": class_names
    }


def save_clustered_results(df, cluster_labels):
    result_df = df.copy()
    result_df["cluster"] = cluster_labels
    result_df.to_csv("clustered_results.csv", index=False)
    print("✅ clustered_results.csv сохранен")


def create_final_report(df, optimal_k, silhouette_scores, binary_results, multiclass_results):
    report = f"""
================================================================================
ОТЧЕТ ПО ПРОЕКТУ ML: АНАЛИЗ ПРОФИЛЕЙ ПОЛЬЗОВАТЕЛЕЙ СТРИМИНГОВОГО СЕРВИСА
================================================================================

1. ОБЗОР ДАННЫХ
────────────────────────────────────────
Всего пользователей: {len(df):,}
Количество признаков: {df.shape[1]}

Средние значения:
- Часы просмотра в месяц: {df['monthly_watch_hours'].mean():.2f}
- Средняя длительность сессии: {df['avg_session_duration'].mean():.2f} минут
- Месяцев подписки: {df['subscription_months'].mean():.2f}
- Дней с последнего входа: {df['days_since_last_login'].mean():.2f}

Распределение типов пользователей:
{df['user_type'].value_counts().to_string()}

2. КЛАСТЕРНЫЙ АНАЛИЗ
────────────────────────────────────────
Оптимальное число кластеров: {optimal_k}

Silhouette Score:
{pd.Series(silhouette_scores).to_string()}

3. БИНАРНАЯ КЛАССИФИКАЦИЯ
────────────────────────────────────────
Задача: прогноз оттока пользователя

Метрики:
- Accuracy: {binary_results['accuracy']:.3f}
- F1-macro: {binary_results['f1']:.3f}
- Precision: {binary_results['precision']:.3f}
- Recall: {binary_results['recall']:.3f}
- ROC-AUC: {binary_results['roc_auc']:.3f}
- Cross-validation F1: {binary_results['cv_f1']:.3f}

Топ-5 важных признаков:
{binary_results['feature_importance'].head(5).to_string(index=False)}

4. МНОГОКЛАССОВАЯ КЛАССИФИКАЦИЯ
────────────────────────────────────────
Задача: определение типа пользователя

Лучшая модель: {multiclass_results['best_model_name']}

Метрики лучшей модели:
- Accuracy: {multiclass_results['best']['accuracy']:.3f}
- F1-macro: {multiclass_results['best']['f1_macro']:.3f}
- Precision: {multiclass_results['best']['precision']:.3f}
- Recall: {multiclass_results['best']['recall']:.3f}

5. КЛЮЧЕВЫЕ ВЫВОДЫ
────────────────────────────────────────
1. Пользователей можно разделить на устойчивые поведенческие сегменты.
2. На риск оттока сильнее всего влияют активность просмотра, количество дней с последнего входа,
   длительность подписки и число обращений в поддержку.
3. Кластеризация помогает выявить группы пользователей с похожим поведением.
4. Модель Random Forest показывает хорошие результаты как в прогнозе оттока,
   так и в классификации типа пользователя.
5. Полученные результаты могут использоваться для персонализации рекомендаций,
   удержания пользователей и оптимизации тарифной политики.

================================================================================
"""
    with open(f"{REPORTS_DIR}/final_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print("✅ reports/final_report.txt сохранен")


def main():
    create_project_structure()
    df = load_data()
    features_df, X_scaled, feature_columns = prepare_features(df)

    plot_correlation_matrix(features_df)

    cluster_labels, optimal_k, silhouette_scores = perform_clustering(features_df, X_scaled)
    save_clustered_results(df, cluster_labels)

    binary_results = perform_binary_classification(features_df, cluster_labels, df)
    multiclass_results = perform_multiclass_classification(features_df, cluster_labels, df)

    create_final_report(df, optimal_k, silhouette_scores, binary_results, multiclass_results)

    print("\n✅ Проект успешно выполнен")
    print("📁 Проверьте папки reports/, reports/figures/, models/")


if __name__ == "__main__":
    main()
