import numpy as np
from sklearn.metrics import accuracy_score
import joblib

# Загрузка предобработанных данных тестового набора
data_test = np.loadtxt("test/data_normal_scaled.txt")  # загружаем предобработанные данные тестового набора

# Загрузка модели
model = joblib.load("trained_model.pkl")

# Загрузка меток классов тестового набора
labels_test = np.loadtxt("test/class_labels.txt")  # загружаем метки классов тестового набора

# Предсказание на тестовых данных
y_pred_test = model.predict(data_test)

# Оценка точности модели на тестовом наборе
accuracy_test = accuracy_score(labels_test, y_pred_test)
print(f"Точность модели на тестовом наборе: {accuracy_test}")