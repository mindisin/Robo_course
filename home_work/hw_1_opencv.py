import cv2
import numpy as np

# Загрузка изображения
# "C:\Users\metro\Desktop\image_opc.jpg"
img = cv2.imread("image_opc.jpg")

if img is None:
    print("Ошибка: Не удалось загрузить изображение.")
    exit()

# Преобразование изображения в градации серого
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Применение пороговой обработки
_, binary_img = cv2.threshold(gray_img, 150, 255, cv2.THRESH_BINARY)

# Использование адаптивной пороговой обработки
adaptive_thresh = cv2.adaptiveThreshold(binary_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Нахождение контуров объектов
contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Подготовка к визуализации
result_img = img.copy()

# Список для хранения площадей объектов
object_areas = []
object_centers = []

# Обработка каждого найденного контура
for idx, contour in enumerate(contours):
    # Расчет площади
    area = cv2.contourArea(contour)
    object_areas.append(area)

    # Моменты для нахождения центра
    moments = cv2.moments(contour)
    if moments["m00"] != 0:
        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])
        object_centers.append((center_x, center_y))

        # Отображение площади рядом с центром объекта
        cv2.putText(result_img, f"{int(area)}", (center_x + 15, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    else:
        print(f"Контур {idx} имеет нулевую площадь")

# Поиск минимального и максимального объектов
if object_areas:
    max_index = np.argmax(object_areas)
    min_index = np.argmin(object_areas)

    # Вывод данных о максимальном объекте
    if len(object_centers) > max_index:
        max_center = object_centers[max_index]
        cv2.circle(result_img, max_center, 5, (0, 255, 0), -1)
        cv2.putText(result_img, f"Max Area: {int(object_areas[max_index])}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        print(f"Центр максимального объекта: {max_center}, площадь: {object_areas[max_index]}")

    # Вывод данных о минимальном объекте
    if len(object_centers) > min_index:
        min_center = object_centers[min_index]
        cv2.circle(result_img, min_center, 5, (255, 0, 0), -1)
        cv2.putText(result_img, f"Min Area: {int(object_areas[min_index])}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        print(f"Центр минимального объекта: {min_center}, площадь: {object_areas[min_index]}")

# Отображение контуров на изображении
cv2.drawContours(result_img, contours, -1, (255, 255, 0), 1)

# Вывод результата
cv2.imshow("Contours and Areas", result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
