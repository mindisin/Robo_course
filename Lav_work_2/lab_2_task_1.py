import cv2
import numpy as np

# Параметры для фильтрации желтого цвета
lower_color = np.array([20, 100, 100])  # Нижняя граница HSV для желтого
upper_color = np.array([30, 255, 255])  # Верхняя граница HSV для желтого

# Открываем видеофайл
video_path = "catball.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Ошибка: не удалось открыть видеофайл.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Завершить, если кадры закончились

    # Преобразуем кадр в HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Маска для выделения желтого цвета
    mask = cv2.inRange(hsv_frame, lower_color, upper_color)
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
    cv2.imshow("Video Processing", frame)

    # Прерывание по клавише 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
