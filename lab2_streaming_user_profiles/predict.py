import joblib
import pandas as pd

FEATURE_COLUMNS = [
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


def predict_user(profile):
    scaler = joblib.load("models/scaler.pkl")
    kmeans = joblib.load("models/kmeans_model.pkl")
    churn_model = joblib.load("models/binary_churn_classifier.pkl")
    user_type_model = joblib.load("models/best_user_type_classifier.pkl")
    encoder = joblib.load("models/user_type_label_encoder.pkl")

    df = pd.DataFrame([profile], columns=FEATURE_COLUMNS)
    scaled = scaler.transform(df)
    cluster = kmeans.predict(scaled)[0]

    df_with_cluster = df.copy()
    df_with_cluster["cluster_label"] = cluster

    churn_prediction = churn_model.predict(df_with_cluster)[0]
    churn_probability = churn_model.predict_proba(df_with_cluster)[0][1]

    user_type_code = user_type_model.predict(df_with_cluster)[0]
    user_type = encoder.inverse_transform([user_type_code])[0]

    return {
        "cluster": int(cluster),
        "churn_prediction": int(churn_prediction),
        "churn_probability": float(churn_probability),
        "user_type": user_type
    }


if __name__ == "__main__":
    example = {
        "age": 24,
        "monthly_watch_hours": 12,
        "avg_session_duration": 25,
        "number_of_sessions": 6,
        "movies_watched": 2,
        "series_watched": 1,
        "documentaries_watched": 0,
        "subscription_months": 3,
        "monthly_fee": 399,
        "support_requests": 4,
        "days_since_last_login": 28,
        "likes_given": 2,
        "ratings_count": 1,
        "watchlist_items": 5,
        "device_count": 1
    }

    result = predict_user(example)

    print("Результат анализа пользователя:")
    print(result)
