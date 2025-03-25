import cv2
from deepface import DeepFace

face_train_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
webcam = cv2.VideoCapture(0)

while True:
    senseful_frame_read, frame = webcam.read()

    if not senseful_frame_read:
        print("Failed to read frame from webcam.")
        break

    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_location = face_train_data.detectMultiScale(grayscale_frame, scaleFactor=1.3, minNeighbors=5)

    if len(face_location) == 0:
        print("No face detected.")
    
    for (x, y, w, h) in face_location:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 100), 2)
        face_roi = frame[y:y + h, x:x + w]
 
        try:
            analysis = DeepFace.analyze(face_roi, actions=['age', 'gender', 'race', 'emotion'], enforce_detection=False)[0]

            age = analysis['age']
            Gender = analysis['gender']
            gender = 'Man' if Gender['Man'] > 50 else 'Woman'
            race = analysis['dominant_race']
            emotion = analysis['dominant_emotion']

            cv2.putText(frame, f"Gender: {gender}", (x, y - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)
            cv2.putText(frame, f"Age: {age}", (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)
            cv2.putText(frame, f"Race: {race}", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)
            cv2.putText(frame, f"Emotion: {emotion}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)

            print(f"Detected: Gender={gender}, Age={age}, Race={race}, Emotion={emotion}")

        except Exception as e:
            print(f"Error analyzing face: {e}")

    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
