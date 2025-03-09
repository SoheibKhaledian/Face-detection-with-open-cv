import cv2


face_trian_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

webcam = cv2.VideoCapture(0)

while True:
    secsefule_fram_read, frame = webcam.read()

    Grayscelad_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_loction = face_trian_data.detectMultiScale(Grayscelad_frame)

    for (x, y, w, h) in face_loction:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (125, 105, 100), 2)

    cv2.imshow('see', frame)
    cv2.waitKey(1)