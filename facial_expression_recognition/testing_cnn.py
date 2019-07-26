import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.model_selection import train_test_split

modelpath = 'model_happy_old.h5'
model = load_model(modelpath)

width, height = 48, 48
data = pd.read_csv('data/fer2013_happy.csv')

# all faces
faces = []
for i in range(data.shape[0]):
    # chuyen doi cot pixels thanh list
    pixel = data['pixels'][0].split()
    # chuyen gia tri str thanh int
    pixel = list(map(int, pixel))
    # reshape 48x48
    face = np.asarray(pixel).reshape(width, height)
    faces.append(face.astype('float32'))
faces = np.asarray(faces)
# them chieu cho moi hinh anh
faces = np.expand_dims(faces, -1)

# chuyen label thanh matran label keras
emotions = pd.get_dummies(data['emotion']).as_matrix()

# chia train, test dataset 90/10
X_train, X_test, y_train, y_test = train_test_split(faces, emotions, test_size=0.1, random_state=42)
# chia train, validation dataset 90/10
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=41)

scores = model.evaluate(np.array(X_test), np.array(y_test), batch_size=64)
print("Loss: {}".format(scores[0]))
print("Accuracy: {}".format(scores[1]))