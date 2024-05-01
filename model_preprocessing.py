import numpy as np
from sklearn.preprocessing import StandardScaler

# Загрузка данных
data = np.loadtxt("train/data_normal.txt")  # загружаем данные из файла

# Применение StandardScaler для предобработки данных
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data.reshape(-1, 1))  # преобразуем данные в формат, подходящий для StandardScaler

# Сохранение предобработанных данных
np.savetxt("train/data_normal_scaled.txt", data_scaled)  # сохраняем предобработанные данные

print("Данные были успешно предобработаны и сохранены.")