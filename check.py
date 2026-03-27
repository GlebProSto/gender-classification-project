import joblib
import pandas as pd

# мужчины и женщины
model = joblib.load('gender_model.pkl')
features = joblib.load('model_features.pkl')

print("✅ Модель загружена!")

new_person = pd.DataFrame({
    'long_hair': [0],
    'forehead_width_cm': [13.5],
    'forehead_height_cm': [6.0],
    'nose_wide': [0],
    'nose_long': [0],
    'lips_thin': [0],
    'distance_nose_to_lip_long': [0]
})

new_person = new_person[features]

prediction = model.predict(new_person)[0]
probability = model.predict_proba(new_person)[0]

if prediction == 1:
    print(f"Пол: Male (уверенность: {probability[1]:.2%})")
else:
    print(f"Пол: Female (уверенность: {probability[0]:.2%})")