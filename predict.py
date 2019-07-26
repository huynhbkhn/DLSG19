import cv2
import numpy as np
from keras.models import load_model

modelpath = 'model_happy.h5'
model = load_model(modelpath)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

emotion_dict = {0: "Smile", 1: "Neutral"}

line_length = 40
line_width = 1

def predict(roi_gray):
    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
    cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
    prediction = model.predict(cropped_img)
    emotion = emotion_dict[int(np.argmax(prediction))]
    return emotion

def draw_border(image, point1, point2, point3, point4, line_length):
    x1, y1 = point1
    x2, y2 = point2
    x3, y3 = point3
    x4, y4 = point4

    cv2.line(image, (x1, y1), (x1, y1 + line_length), (0, 255, 0), line_width)  # -- top-left
    cv2.line(image, (x1, y1), (x1 + line_length, y1), (0, 255, 0), line_width)

    cv2.line(image, (x2, y2), (x2, y2 - line_length), (0, 255, 0), line_width)  # -- bottom-left
    cv2.line(image, (x2, y2), (x2 + line_length, y2), (0, 255, 0), line_width)

    cv2.line(image, (x3, y3), (x3 - line_length, y3), (0, 255, 0), line_width)  # -- top-right
    cv2.line(image, (x3, y3), (x3, y3 + line_length), (0, 255, 0), line_width)

    cv2.line(image, (x4, y4), (x4, y4 - line_length), (0, 255, 0), line_width)  # -- bottom-right
    cv2.line(image, (x4, y4), (x4 - line_length, y4), (0, 255, 0), line_width)

    return image

def facedetection(image, faceBb, emotionLbl):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    smile = False
    for (x, y, w, h) in faces:
        # cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 1)
        if faceBb:
            point1, point2, point3, point4 = (x, y), (x, y + h), (x + w, y), (x + w, y + h)
            image = draw_border(image, point1, point2, point3, point4, line_length)

        roi_gray = gray[y:y + h, x:x + w]
        emotion = predict(roi_gray)
        if emotionLbl:
            cv2.putText(image, emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        if emotion == "Smile":
            smile = True

    return image, smile