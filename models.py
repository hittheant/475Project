from keras import models
from keras import layers
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class MaskClassifier():
    def __init__(self):
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(30, 30, 3)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(225, activation='sigmoid'))
        self.model.add(layers.Dense(3, activation='softmax'))
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

    def train(self, x_train, y_train, x_test, y_test):
        print(self.model.summary())
        x_train = np.array(x_train)/255.0
        x_test = np.array(x_test)/255.0
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=3)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes=3)
        history = self.model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')
        plt.show()

        test_loss, test_acc = self.model.evaluate(x_test, y_test, verbose=2)
