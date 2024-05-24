from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier(r'D:\abhijith\ML\pravaah\models\emotion\haarcascade_frontalface_default.xml')
classifier = load_model(r'D:\abhijith\ML\pravaah\models\emotion\model.h5')

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

video_path = r"D:\abhijith\ML\pravaah\video.avi"  # Change this to the path of your video file
cap = cv2.VideoCapture(video_path)

emotion_counts = {label: 0 for label in emotion_labels}  # Initialize counts to zero

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            emotion_counts[label] += 1  # Increment count for detected emotion
        else:
            pass

print("Emotion Counts:")
for emotion, count in emotion_counts.items():
    print(f"{emotion}: {count}")

cap.release()
cv2.destroyAllWindows()
