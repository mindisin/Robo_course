import cv2
import numpy as np
import matplotlib.pyplot as plt

# Загрузка изображения
image = cv2.imread("snakes.jpg")
if image is None:
    print("Ошибка: Не удалось загрузить изображение.")
    exit()

# Ядро для повышения резкости
kernel = np.array([[0, -1, 0], 
                   [-1, 5, -1],
                   [0, -1, 0]])

# Применяем фильтр для повышения резкости
sharpened = cv2.filter2D(image, -1, kernel)

# 1. Гауссово размытие с увеличенным ядром
blurred_image = cv2.GaussianBlur(image, (15, 15), 0)

# 2. Выделение границ с использованием оператора Собеля
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Преобразование в градации серого
sobel_edges = cv2.Sobel(gray_image, cv2.CV_64F, 1, 1, ksize=5)  # Градиенты по X и Y
edges = cv2.convertScaleAbs(sobel_edges)  # Преобразование в 8-битный формат

# Преобразование границ в трёхканальное изображение (выполнено ДО комбинирования)
edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

# 3. Комбинирование изображений
combined_1 = cv2.addWeighted(blurred_image, 0.5, edges_color, 0.5, 0)
combined_final = cv2.addWeighted(combined_1, 0.6, sharpened, 0.4, 0)

# Отображение результатов с использованием Matplotlib
def show_images(original, blurred, edges, sharpened, combined):
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 3, 1)
    plt.title('Оригинальное изображение')
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title('Гауссово размытие')
    plt.imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title('Выделение границ (Собель)')
    plt.imshow(edges, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.title('Повышение резкости (с использованием ядра)')
    plt.imshow(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.title('Комбинированное изображение')
    plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()
    plt.show()

show_images(image, blurred_image, edges, sharpened, combined_final)
