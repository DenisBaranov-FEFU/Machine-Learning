import os
import joblib


MODEL_PATH = "models/model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"


def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Модель не найдена. Сначала запустите: py code.py")

    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError("Vectorizer не найден. Сначала запустите: py code.py")

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer


def predict_category(text):
    model, vectorizer = load_artifacts()
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]
    probabilities = model.predict_proba(text_vector)[0]

    classes = model.classes_
    confidence = max(probabilities)

    return prediction, confidence, dict(zip(classes, probabilities))


if __name__ == "__main__":
    print("Введите обращение пользователя. Для выхода напишите exit.\n")

    while True:
        user_text = input("Сообщение: ")

        if user_text.lower() in ["exit", "quit", "выход"]:
            break

        category, confidence, probabilities = predict_category(user_text)

        print(f"\nКатегория: {category}")
        print(f"Уверенность: {confidence:.2f}")
        print("Вероятности по классам:")

        for cls, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
            print(f"- {cls}: {prob:.2f}")

        print()
