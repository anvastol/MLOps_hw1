#!/bin/bash

# Запуск скрипта model_preparation.py для создания и обучения модели
echo "Запуск скрипта model_preparation.py..."
python model_preparation.py
echo "Скрипт model_preparation.py завершен."

# Запуск скрипта model_testing.py для проверки модели на тестовых данных
echo "Запуск скрипта model_testing.py..."
python model_testing.py
echo "Скрипт model_testing.py завершен."