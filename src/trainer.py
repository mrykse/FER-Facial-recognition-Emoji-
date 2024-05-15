import os
import sys
import logging
import tensorflow as tf
import numpy as np
import tqdm
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
from src.data_manager import EmojifierDataManager
from src.__init__ import *

logger = logging.getLogger('emojifier.model')

batch_size = 64
train_steps = 1000
learning_rate = 0.001
dropout_keep_prob = 0.8


def weight_variable(shape):
    initial = tf.random.truncated_normal(shape=shape, stddev=0.1)
    return initial


def bias_variable(shape):
    initial = tf.random.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial)


class ConvLayer(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size):
        super(ConvLayer, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='same')
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.Activation('relu')

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.batch_norm(x)
        x = self.activation(x)
        return x

def build_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        # Convolutional layers
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        # Flatten layer
        Flatten(),

        # Fully connected layers
        Dense(128, activation='relu'),

        # Output layer
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def test(emoji_data, model):
    logger.info('CALCULATING TESTSET ACCURACY ...')
    L = len(emoji_data.test.labels)

    x = emoji_data.test.images
    y = emoji_data.test.labels

    accs = []

    for i in tqdm.tqdm(range(0, L, 30)):
        if i + 30 <= L:
            x_i = x[i:i + 30].reshape(30, 48, 48, 1)
            y_i = y[i:i + 30].reshape(30, len(EMOTION_MAP))
        else:
            x_i = x[i:].reshape(L - i, 48, 48, 1)
            y_i = y[i:].reshape(L - i, len(EMOTION_MAP))

        predictions = model.predict(x_i)
        acc = np.mean(np.argmax(predictions, axis=1) == np.argmax(y_i, axis=1))
        accs.append(acc)

    acc = np.mean(accs)

    logger.critical('test-accuracy: {:.4}%'.format(acc * 100))


if __name__ == '__main__':
    CHECKPOINT_SAVE_PATH = os.path.join(os.path.dirname(__file__), os.pardir, 'model_checkpoints')

    if not os.path.exists(CHECKPOINT_SAVE_PATH):
        os.makedirs(CHECKPOINT_SAVE_PATH)

    BATCH_SIZE = config_parser.getint('MODEL_HYPER_PARAMETERS', 'batch_size')
    STEPS = config_parser.getint('MODEL_HYPER_PARAMETERS', 'train_steps')
    LEARNING_RATE = config_parser.getfloat('MODEL_HYPER_PARAMETERS', 'learning_rate')
    KEEP_PROB = config_parser.getfloat('MODEL_HYPER_PARAMETERS', 'dropout_keep_prob')

    emoset = EmojifierDataManager()

    logger.info("Number of train images: {}".format(len(emoset.train.images)))
    logger.info("Number of train labels: {}".format(len(emoset.train.labels)))
    logger.info("Number of test images: {}".format(len(emoset.test.images)))
    logger.info("Number of test labels: {}".format(len(emoset.test.labels)))

    input_shape = (48, 48, 1)
    num_classes = len(EMOTION_MAP)

    model = build_model(input_shape, num_classes)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    for i in tqdm.tqdm(range(STEPS)):
        x_data, y_data = emoset.train.next_batch(BATCH_SIZE)

        model.fit(x_data, y_data, batch_size=BATCH_SIZE, epochs=1, verbose=0)

        if i % 20 == 0:
            acc, loss = model.evaluate(x_data, y_data, verbose=0)
            logger.info('accuracy: {:.4}%, loss: {:.4}'.format(acc * 100, loss))

    test(emoset, model)

    model.save(os.path.join(CHECKPOINT_SAVE_PATH, 'model.h5'))
    logger.info("Model saved in path: {}".format(os.path.join(CHECKPOINT_SAVE_PATH, 'model.h5')))
