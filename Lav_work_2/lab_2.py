import cv2
import numpy as np

# **Общие настройки**
# Диапазон цвета мяча (желтый)
YELLOW_LOWER = np.array([20, 100, 100])
YELLOW_UPPER = np.array([30, 255, 255])

# Минимальная площадь контура для исключения шумов
MIN_CONTOUR_AREA = 500

# Функция обработки одного кадра
def process_frame(frame, lower_color, upper_color):
    """
    Обрабатывает один кадр: детектирует объект, рисует контуры и выводит центр масс.

    :param frame: Кадр для обработки
    :param lower_color: Нижняя граница цвета в HSV
    :param upper_color: Верхняя граница цвета в HSV
    :return: Обработанный кадр
    """
    # Преобразование в цветовое пространство HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Маска по цвету
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Обнаружение контуров
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Выбор самого большого контура
        largest_contour = max(contours, key=cv2.contourArea)

        # Проверка на минимальную площадь
        if cv2.contourArea(largest_contour) > MIN_CONTOUR_AREA:
            # Рисование контура
            cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)

            # Вычисление центра масс
            M = cv2.moments(largest_contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

                # Вывод координат центра масс
                cv2.putText(frame, f"Center: ({cx}, {cy})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "No object detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    else:
        cv2.putText(frame, "No object detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return frame

# **Обработка видеофайла**
def process_video(file_path):
    cap = cv2.VideoCapture(file_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Обработка кадра
        processed_frame = process_frame(frame, YELLOW_LOWER, YELLOW_UPPER)

        # Отображение результата
        cv2.imshow('Processed Video', processed_frame)

        # Выход по нажатию клавиши 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# **Обработка видеопотока с камеры**
def process_camera():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Обработка кадра
        processed_frame = process_frame(frame, YELLOW_LOWER, YELLOW_UPPER)

        # Отображение результата
        cv2.imshow('Camera Feed', processed_frame)

        # Выход по нажатию клавиши 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# **Запуск обработки**
if __name__ == "__main__":
    print("Processing video file...")
    process_video("catball.mp4")

    print("Switching to camera feed...")
    process_camera()
