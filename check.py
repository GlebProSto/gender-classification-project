from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Пути к сохранённой модели и списку признаков.
MODEL_FILE = Path("gender_model.pkl")
FEATURES_FILE = Path("model_features.pkl")

# Добавляем к новым данным тот же признак, что использовался при обучении.
def add_features(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["forehead_ratio"] = (
        result["forehead_width_cm"]
        / result["forehead_height_cm"].replace(0, np.nan)
    )
    result["forehead_ratio"] = result["forehead_ratio"].fillna(0)
    return result

# Загружаем обученную модель и порядок признаков.
model = joblib.load(MODEL_FILE)
features = joblib.load(FEATURES_FILE)
print("Модель загружена.")
# Задаём признаки одного нового человека для проверки.
new_person = pd.DataFrame(
    {
        "long_hair": [0],
        "forehead_width_cm": [13.5],
        "forehead_height_cm": [6.0],
        "nose_wide": [0],
        "nose_long": [0],
        "lips_thin": [0],
        "distance_nose_to_lip_long": [0],
    }
)
new_person = add_features(new_person)[features]
# Получаем вероятность класса Male и общую уверенность модели.
probability_male = model.predict_proba(new_person)[0, 1]
confidence = max(probability_male, 1 - probability_male)
# Для пограничной вероятности сообщаем, что модель не уверена.
if 0.4 <= probability_male <= 0.6:
    print(
        f"Результат: модель не уверена "
        f"(вероятность Male: {probability_male:.1%})."
    )
elif probability_male > 0.6:
    print(f"Предсказание: Male (уверенность: {confidence:.1%})")
else:
    print(f"Предсказание: Female (уверенность: {confidence:.1%})")
