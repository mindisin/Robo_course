import cv2
import glob
import numpy as np
import os

# Размеры шахматной доски
chessboard_size = (9, 6)
square_size = 1.5

# Создаем объект для углов шахматной доски
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Массивы для хранения точек
objpoints = []  # Мировые координаты точек
imgpoints = []  # Координаты точек на изображении

# Загружаем изображения для калибровки
images = glob.glob('chessboard_images/*.jpg')

# Обрабатываем каждое изображение
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Ищем углы шахматной доски
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    
    if ret:
        objpoints.append(objp)  # Добавляем мировые координаты
        imgpoints.append(corners)  # Добавляем координаты на изображении

# Калибруем камеру, вычисляя матрицу камеры и искажения
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Выводим параметры калибровки
print("Калибровка завершена:")
print(f"Камера матрица (mtx): \n{mtx}")
print(f"Искажения (dist): \n{dist}")
print(f"Фокусное расстояние по оси X (fx): {mtx[0][0]}")
print(f"Фокусное расстояние по оси Y (fy): {mtx[1][1]}")

# Извлекаем фокусные расстояния
fx = mtx[0][0]
fy = mtx[1][1]

# Матрица камеры
print("Матрица камеры: ", mtx)

# Задержка для отображения или сохранения результата
