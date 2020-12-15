from keras import models
from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class MaskClassifier():
    def __init__(self):
        self.model = Sequential()
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(30, 30, 3)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(225, activation='sigmoid'))
        self.model.add(layers.Dense(3, activation='softmax'))
        self.model.compile(optimizer=Adam(lr=0.0005),
                           loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

    def train(self, x_train, y_train, x_test, y_test):
        print(self.model.summary())
        tf.keras.utils.plot_model(
            self.model,
            to_file="./model.png",
            show_shapes=False,
            show_layer_names=True
        )
        x_train = np.array(x_train)/255.0
        x_test = np.array(x_test)/255.0
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=3)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes=3)
        history = self.model.fit(x_train, y_train, epochs=18, validation_data=(x_test, y_test))
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_accuracy'], label='validation accuracy')
        plt.plot(history.history['val_loss'], label='validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')
        plt.savefig('./results/training.png')

        xs = np.load('./archive/raw_splits/x_test.npy')
        ys = np.load('./archive/raw_splits/y_test.npy')
        y_hats = self.model.predict(xs.astype(np.float32))
        np.save('./results/y_test_full_pred.npy', y_hats)
        np.save('./results/y_test_full.npy', ys)
        y_train_pred = self.model.predict(x_train)
        y_pred = self.model.predict(x_test)
        np.save('./results/y_train_pred.npy', y_train_pred)
        np.save('./results/y_train.npy', y_train)
        np.save('./results/y_test_pred.npy', y_pred)
        np.save('./results/y_test.npy', y_test)
        np.save('./results/x_test.npy', x_test)
        # self.model.save('./results/m')
        test_loss, test_acc = self.model.evaluate(x_test, y_test, verbose=2)
        print('\n', test_loss, test_acc)
