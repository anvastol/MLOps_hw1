import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Загрузка предобработанных данных
data = np.loadtxt("train/data_normal_scaled.txt")  # загружаем предобработанные данные

# Загрузка меток классов
labels = np.loadtxt("train/class_labels.txt")  # загружаем метки классов

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Создание модели (в данном случае RandomForestClassifier)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Обучение модели на обучающих данных
model.fit(X_train, y_train)

# Предсказание на тестовых данных
y_pred = model.predict(X_test)

# Оценка точности модели
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели: {accuracy}")

# Сохранение обученной модели
joblib.dump(model, "trained_model.pkl")
print("Модель была успешно обучена и сохранена.")