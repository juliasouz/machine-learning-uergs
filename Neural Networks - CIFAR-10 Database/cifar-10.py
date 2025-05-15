import keras
from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import tensorflow as tf

# Loading the CIFAR-10 Dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print("Training data shape:", x_train.shape)

# Displaying 16 Images from the Training Set
plt.figure(figsize=(6, 6))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(x_train[i + 20])
    plt.axis('off')
plt.tight_layout()
plt.show()

# Checking the Shape of the Training Labels
print("Labels shape:", y_train.shape)

# One-Hot Encoding the Labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Normalizing the Images
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Building the Convolutional Neural Network (CNN) Model
cnn_model = Sequential()
cnn_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3))) 
cnn_model.add(MaxPooling2D((2, 2))) 
cnn_model.add(Conv2D(64, (3, 3), activation='relu')) 
cnn_model.add(MaxPooling2D((2, 2)))
cnn_model.add(Flatten())
cnn_model.add(Dense(512, activation='relu')) 
cnn_model.add(Dropout(0.5))
cnn_model.add(Dense(10, activation='softmax')) 

# Compiling and Training the Model
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = cnn_model.fit(x_train, y_train, batch_size=200, epochs=32, verbose=1, validation_data=(x_test, y_test))

# Evaluating the Model on the Test Set
test_loss, test_accuracy = cnn_model.evaluate(x_test, y_test)
print('Loss:', test_loss)
print('Accuracy:', test_accuracy)

# Plotting Training and Validation Loss
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plotting Training and Validation Accuracy
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Predicting the Class of a New Image
img = keras.preprocessing.image.load_img('image.jpeg', target_size=(32, 32))
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)
classes = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Making a Prediction and Displaying the Result
predictions = cnn_model.predict(img_array)
score = tf.nn.softmax(predictions[0])
print("Predicted class:", classes[np.argmax(score)])
print("Confidence: {:.2f}%".format(100 * np.max(score)))

# Edge Detection on the Image Using Canny Algorithm
import cv2
img_cv = cv2.imread('image.jpeg', 0)
edges = cv2.Canny(img_cv, 100, 200)
plt.subplot(1, 2, 1)
plt.imshow(img_cv, cmap='gray')
plt.title('Original Image')
plt.xticks([]), plt.yticks([])
plt.subplot(1, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title('Detected Edges')
plt.xticks([]), plt.yticks([])
plt.show()