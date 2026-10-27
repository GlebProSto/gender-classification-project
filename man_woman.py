from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import export_text


RANDOM_STATE = 42
DATA_FILE = Path("gender_classification_v7.csv")
MODEL_FILE = Path("gender_model.pkl")
FEATURES_FILE = Path("model_features.pkl")

# Создаём дополнительный признак: отношение ширины лба к его высоте.
def add_features(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["forehead_ratio"] = (
        result["forehead_width_cm"]
        / result["forehead_height_cm"].replace(0, np.nan)
    )
    result["forehead_ratio"] = result["forehead_ratio"].fillna(0)
    return result

# Загружаем таблицу и переводим Male/Female в числа 1/0.
df = pd.read_csv(DATA_FILE)
df["gender"] = (df["gender"] == "Male").astype(int)
# Отделяем признаки от правильных ответов модели.
X = add_features(df.drop(columns="gender"))
y = df["gender"]
features = X.columns.tolist()
# Делим данные на обучение, проверку выбора модели и финальный тест.
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=RANDOM_STATE
)
X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_STATE
)
# Подбираем параметры Random Forest по пяти фолдам кросс-валидации.
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10, 15],
    "min_samples_leaf": [1, 3, 5, 10],
}
grid_search = GridSearchCV(
    RandomForestClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    ),
    param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
)
grid_search.fit(X_train, y_train)
# Создаём несколько моделей для честного сравнения на одинаковых данных.
models = {
    "RandomForest": grid_search.best_estimator_,
    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    ),
    "HistGradientBoosting": HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=15,
        random_state=RANDOM_STATE,
    ),
    "LogisticRegression": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]
    ),
}
# Обучаем каждую модель и собираем основные метрики на validation.
validation_results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    prediction = model.predict(X_valid)
    probability = model.predict_proba(X_valid)[:, 1]
    validation_results.append(
        {
            "model": name,
            "accuracy": accuracy_score(y_valid, prediction),
            "balanced_accuracy": balanced_accuracy_score(y_valid, prediction),
            "f1": f1_score(y_valid, prediction),
            "roc_auc": roc_auc_score(y_valid, probability),
        }
    )
# Выбираем модель с лучшим ROC-AUC.
results = pd.DataFrame(validation_results).sort_values(
    "roc_auc", ascending=False
)
print("=== Сравнение моделей на validation ===")
print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

best_name = results.iloc[0]["model"]
best_model = models[best_name]
print(f"\nВыбрана модель: {best_name}")
print(f"Лучшие параметры RandomForest: {grid_search.best_params_}")

# Дообучаем выбранную модель на train + validation и калибруем вероятности.
X_final = pd.concat([X_train, X_valid])
y_final = pd.concat([y_train, y_valid])
calibrated_model = CalibratedClassifierCV(
    estimator=best_model,
    method="sigmoid",
    cv=5,
)
calibrated_model.fit(X_final, y_final)
# Проверяем окончательное качество только на ранее неиспользованном test.
predict_test = calibrated_model.predict(X_test)
probability_test = calibrated_model.predict_proba(X_test)[:, 1]
print("\n=== Итоговая оценка на test ===")
print(classification_report(y_test, predict_test, target_names=["Female", "Male"]))
print(f"Accuracy: {accuracy_score(y_test, predict_test):.4f}")
print(f"Balanced accuracy: {balanced_accuracy_score(y_test, predict_test):.4f}")
print(f"F1: {f1_score(y_test, predict_test):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, probability_test):.4f}")
print("Матрица ошибок:")
print(confusion_matrix(y_test, predict_test))
# Для древовидной модели показываем вклад каждого признака.
if hasattr(best_model, "feature_importances_"):
    feature_importance = pd.DataFrame(
        {"feature": features, "importance": best_model.feature_importances_}
    ).sort_values("importance", ascending=False)
    print("\n=== Важность признаков ===")
    print(feature_importance.to_string(index=False))

    plt.figure(figsize=(10, 6))
    plt.barh(
        feature_importance["feature"],
        feature_importance["importance"],
        color="steelblue",
    )
    plt.xlabel("Важность признака")
    plt.title(f"Важность признаков ({best_name})")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.close()

if best_name == "RandomForest":
    print("\n=== Правила первого дерева ===")
    print(export_text(best_model.estimators_[0], feature_names=features, max_depth=3))
# Сохраняем модель и точный порядок признаков для check.py.
joblib.dump(calibrated_model, MODEL_FILE)
joblib.dump(features, FEATURES_FILE)
print(f"\nМодель сохранена: {MODEL_FILE}")
print(f"Список признаков сохранён: {FEATURES_FILE}")
