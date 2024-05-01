import numpy as np
import os

# Создание данных без аномалий
def create_data_normal(num_samples):
    temperature = np.random.randint(10, 30, num_samples)
    return temperature

# Создание данных с аномалиями
def create_data_anomalies(num_samples):
    temperature = np.random.randint(10, 30, num_samples)
    temperature[5] = 100  # добавляем аномалию
    return temperature

# Создание данных с шумами
def create_data_noise(num_samples):
    temperature = np.random.randint(10, 30, num_samples)
    noise = np.random.normal(0, 5, num_samples)
    temperature_with_noise = temperature + noise
    return temperature_with_noise

# Создание папок "train" и "test"
os.makedirs("train", exist_ok=True)
os.makedirs("test", exist_ok=True)

# Генерация данных и сохранение их в папки "train" и "test"
num_samples = 30

data_normal = create_data_normal(num_samples)
np.savetxt("train/data_normal.txt", data_normal[:20])  # сохраняем первые 20 значений для тренировки
np.savetxt("test/data_normal.txt", data_normal[20:])  # сохраняем оставшиеся для тестирования

data_anomalies = create_data_anomalies(num_samples)
np.savetxt("train/data_anomalies.txt", data_anomalies[:20])
np.savetxt("test/data_anomalies.txt", data_anomalies[20:])

data_noise = create_data_noise(num_samples)
np.savetxt("train/data_noise.txt", data_noise[:20])
np.savetxt("test/data_noise.txt", data_noise[20:])

print("Наборы данных успешно созданы и сохранены в папках 'train' и 'test'.")