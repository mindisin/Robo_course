import cv2
import numpy as np

# Параметры для фильтрации красного цвета
# Первый диапазон красного (от 0 до 10)
lower_color1 = np.array([0, 120, 70])  # Нижняя граница HSV для красного
upper_color1 = np.array([10, 255, 255])  # Верхняя граница HSV для красного

# Второй диапазон красного (от 170 до 180)
lower_color2 = np.array([170, 120, 70])  # Нижняя граница HSV для красного
upper_color2 = np.array([180, 255, 255])  # Верхняя граница HSV для красного

# Открываем поток с камеры
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Ошибка: не удалось открыть камеру.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Ошибка: не удалось получить кадр.")
        break

    # Преобразуем кадр в HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Маска для выделения красного цвета (оба диапазона)
    mask1 = cv2.inRange(hsv_frame, lower_color1, upper_color1)
    mask2 = cv2.inRange(hsv_frame, lower_color2, upper_color2)

    # Объединяем оба диапазона красного цвета
    mask = cv2.bitwise_or(mask1, mask2)
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # Найти контуры
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Выбираем самый большой контур
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 500:  # Условие на минимальную площадь
            # Рисуем контур
            cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)

            # Находим центр масс
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)  # Отображаем центр масс
                text = f"Center: ({cx}, {cy})"
            else:
                text = "No object detected"
        else:
            text = "Object too small"
    else:
        text = "No object detected"

    # Выводим текст на экран
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Отображаем результат
    cv2.imshow("Camera Processing", frame)

    # Прерывание по клавише 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
