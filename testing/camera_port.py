import cv2

for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Caméra trouvée à l’index {i}")
    cap.release()
