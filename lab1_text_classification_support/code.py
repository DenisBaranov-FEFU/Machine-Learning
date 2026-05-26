import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


DATA_PATH = "data.csv"
REPORTS_DIR = "reports"
MODELS_DIR = "models"


def create_dirs():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)


def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл {path} не найден")

    df = pd.read_csv(path)

    required_columns = {"text", "category"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"В датасете отсутствуют колонки: {missing}")

    df = df.dropna(subset=["text", "category"])
    df["text"] = df["text"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()
    df = df[df["text"] != ""]

    if len(df) < 50:
        print("⚠️ Внимание: датасет маленький, метрики могут быть нестабильными.")

    return df


def train_model(df):
    X = df["text"]
    y = df["category"]

    print("📊 Количество строк:", len(df))
    print("📊 Количество классов:", y.nunique())
    print("\n📌 Распределение классов:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        analyzer="word",
        ngram_range=(1, 2),
        max_features=7000
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    print("\n🧠 Обучение модели...")
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    print("\n✅ Accuracy:", round(accuracy, 4))
    print("\n✅ Classification report:")
    print(report)

    return model, vectorizer, y_test, y_pred, accuracy, report


def save_results(model, vectorizer, y_test, y_pred, accuracy, report):
    with open(os.path.join(REPORTS_DIR, "report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write(report)

    labels = sorted(list(set(y_test) | set(y_pred)))
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    plt.figure(figsize=(11, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )
    plt.xlabel("Предсказано")
    plt.ylabel("Реально")
    plt.title("Матрица ошибок классификации обращений")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "confusion_matrix.png"), dpi=300)
    plt.close()

    joblib.dump(model, os.path.join(MODELS_DIR, "model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "vectorizer.pkl"))

    print("\n💾 Сохранено:")
    print("- reports/report.txt")
    print("- reports/confusion_matrix.png")
    print("- models/model.pkl")
    print("- models/vectorizer.pkl")


def main():
    create_dirs()
    df = load_data(DATA_PATH)
    model, vectorizer, y_test, y_pred, accuracy, report = train_model(df)
    save_results(model, vectorizer, y_test, y_pred, accuracy, report)


if __name__ == "__main__":
    main()
