import pandas as pd
import numpy as np
from keras import utils
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, Flatten


train_df = pd.read_csv("train.csv")

y = train_df["label"].copy()
X = train_df.drop("label", axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_train_enc = utils.to_categorical(y_train, num_classes=10)
y_test_enc = utils.to_categorical(y_test, num_classes=10)

X_train = np.asarray(X_train).reshape((-1, 28, 28)) / 255.0
X_test = np.asarray(X_test).reshape((-1, 28, 28)) / 255.0

model = Sequential()
model.add(Conv1D(128, kernel_size=5, input_shape=(28,28), activation='relu'))
model.add(Conv1D(128, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Dropout(0.25))

model.add(Conv1D(256, kernel_size=5, activation='relu'))
model.add(Conv1D(256, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(X_train, y_train_enc, batch_size=256, epochs=20, validation_data=(X_test, y_test_enc))

# # # # # # # # # # # # # #

test_df = pd.read_csv('test.csv')
test_data = np.asarray(test_df).reshape((-1, 28, 28)) / 255.0
predicted_classes = model.predict_classes(test_data)

results = pd.Series(predicted_classes, name="Label")
submission = pd.concat([pd.Series(range(1, 28001), name="ImageId"), results], axis = 1)
submission.to_csv("cnn_mnist_submission.csv",index=False)
