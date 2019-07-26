import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping, ModelCheckpoint

width, height = 48, 48

data = pd.read_csv('data/fer2013_happy.csv')

pixels = data['pixels'].tolist()  # 1

# all faces
faces = []
for i in range(data.shape[0]):
    # chuyen doi cot pixels thanh list
    pixel = data['pixels'][i].split()
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

"""
Cau truc mang

    [2 x CONV (3x3)] — MAXP (2x2) — DROPOUT (0.5)
    [2 x CONV (3x3)] — MAXP (2x2) — DROPOUT (0.5)
    [2 x CONV (3x3)] — MAXP (2x2) — DROPOUT (0.5)
    [2 x CONV (3x3)] — MAXP (2x2) — DROPOUT (0.5)
    Dense (512) — DROPOUT (0.5)
    Dense (256) — DROPOUT (0.5)
    Dense (128) — DROPOUT (0.5)
"""

num_labels = 2
epochs = 100
num_features, batch_size = 64, 64
modelpath = 'model_happy.h5'

model = Sequential()

model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), data_format='channels_last', kernel_regularizer=l2(0.01)))
model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(2*2*2*num_features, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2*2*num_features, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2*num_features, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(num_labels, activation='softmax'))

print(model.summary())

model.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])

lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, verbose=1)
tensorboard = TensorBoard(log_dir='./logs')
early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=200, verbose=1, mode='auto')
checkpointer = ModelCheckpoint(modelpath, monitor='val_loss', verbose=1, save_best_only=True)


seqModel = model.fit(np.array(X_train), np.array(y_train),
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(np.array(X_test), np.array(y_test)),
          shuffle=True,
          callbacks=[lr_reducer, tensorboard, early_stopper, checkpointer])

scores = model.evaluate(np.array(X_test), np.array(y_test), batch_size=batch_size)
print("Loss: {}".format(scores[0]))
print("Accuracy: {}".format(scores[1]))

# visualizing losses and accuracy
train_loss = seqModel.history['loss']
val_loss   = seqModel.history['val_loss']
train_acc  = seqModel.history['acc']
val_acc    = seqModel.history['val_acc']
xc         = range(epochs)

plt.subplot(1, 2, 1)
plt.plot(xc, train_loss, label='train_loss')
plt.plot(xc, val_loss, label='val_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(xc, train_acc, label='train_acc')
plt.plot(xc, val_acc, label='val_acc')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.legend()

plt.show()

