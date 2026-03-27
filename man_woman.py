import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV # GridSearchCV — это универсальный инструмент. 
# Ему всё равно, какая внутри модель. Главное, чтобы модель следовала правилам scikit-learn (имела методы .fit() и .predict()).
import joblib

df = pd.read_csv('gender_classification_v7.csv')
df['gender'] = (df['gender'] == 'Male').astype(int) 
    
features = df.columns.drop('gender').tolist() # Создает список имен всех колонок, кроме целевой ('gender')

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 8)) # Создает поле для графиков: 2 ряда, 4 колонки (всего 8 мест). 
axes = axes.flatten() # flatten() превращает сетку в простой список осей одномерный, чтобы удобно перебирать их в цикле.

for i, col in enumerate(features):
    ax = axes[i]
    
    males = df[df['gender'] == 1][col]
    females = df[df['gender'] == 0][col]
    
    if df[col].nunique() == 2 and df[col].isin([0, 1]).all():
        bins = [-0.5, 0.5, 1.5]
        ax.hist(females, bins=bins, color='red', alpha=0.7, density=True, label='Female', edgecolor='white')
        ax.hist(males, bins=bins, color='blue', alpha=0.7, density=True, label='Male', edgecolor='white')
    else:
        ax.hist(females, bins=20, color='red', alpha=0.7, density=True, label='Female', edgecolor='white')
        ax.hist(males, bins=20, color='blue', alpha=0.7, density=True, label='Male', edgecolor='white')
    ax.set_title(col, fontsize=12)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_xlabel(col, fontsize=10)
    ax.legend(fontsize=9)

axes[-1].axis('off')
plt.tight_layout()
plt.show()

train,valid,test = np.split(df.sample(frac=1),[int(0.6*len(df)), int(0.8*len(df))])
# df.sample(frac=1): Перемешивает все строки в случайном порядке (очень важно, чтобы данные не шли по порядку).
X_train = train.drop('gender',axis=1)
y_train = train['gender']

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, 10],
    'min_samples_leaf': [5, 10, 20]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42), 
    param_grid, 
    cv=5, 
    scoring='accuracy',
    n_jobs=-1  # Использовать все ядра процессора для скорости
) # GridSearchCV: Перебирает все комбинации из param_grid.
# cv=5: Использует кросс-валидацию (разбивает Train на 5 частей, 5 раз учит и проверяет), чтобы оценка была надежной.

grid_search.fit(X_train, y_train)
print(f"Лучшие параметры: {grid_search.best_params_}")
print(f"Лучшая точность: {grid_search.best_score_:.3f}")

clf = grid_search.best_estimator_ 
clf.fit(X_train, y_train)

X_valid = valid.drop('gender', axis=1)
y_valid = valid['gender']

X_test = test.drop('gender', axis=1)
y_test = test['gender']

predict_valid = clf.predict(X_valid) 
predict_test = clf.predict(X_test)
print(classification_report(y_test, predict_test, target_names=['Female', 'Male']))


accuracy_valid = accuracy_score(y_valid, predict_valid)
print(f"Точность на валидации: {accuracy_valid:.2f} ({accuracy_valid*100:.1f}%)")

accuracy_test = accuracy_score(y_test, predict_test)
print(f"Точность на тесте: {accuracy_test:.2f} ({accuracy_test*100:.1f}%)")


print("\n=== Матрица ошибок (Тест) ===")
cm = confusion_matrix(y_test, predict_test)
print(cm)


print("\n=== Подробный отчёт (Тест) ===")
print(classification_report(y_test, predict_test, target_names=['Female', 'Male']))

plt.figure(figsize=(12, 6))
importances = clf.feature_importances_
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': importances
}).sort_values('importance', ascending=False)

plt.barh(feature_importance['feature'], feature_importance['importance'], color='steelblue')
plt.xlabel('Важность признака')
plt.title('Важность признаков (RandomForest)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

print("\n=== Важность признаков ===")
print(feature_importance)

print("\n=== Правила первого дерева из 200 ===")
tree_rules = export_text(clf.estimators_[0], feature_names=features, max_depth=3)
print(tree_rules)

joblib.dump(clf, 'gender_model.pkl')
print("\n✅ Модель сохранена в файл: gender_model.pkl")

joblib.dump(features, 'model_features.pkl') # Сохраняя features, вы фиксируете эталон.То есть признаки в правильном порядке
print("✅ Список признаков сохранён: model_features.pkl")