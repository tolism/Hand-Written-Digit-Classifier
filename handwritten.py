'''
 FILE: handwritten.py
 Convolution Neural Network for MNIST Handwritten digit classification
 Using Tensorflow 2
 Author:
   Moustaklis Apostolos , amoustakl@auth.gr
'''

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold

# Checking tf version ( 2.1.0 )
print(tf.__version__)


def load_dataset():
    # Get the training / testing data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Summarize loaded dataset
    print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
    print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))

    # Plot some images
    for i in range(9):
        # define subplot
        plt.subplot(330 + 1 + i)
        # plot raw pixel data
        plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))
    # show the figure
    plt.show()
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    return x_train, y_train, x_test, y_test


def rescale_photo(train, test):
    # Rescaling the train / test images from [0 , 255 ] to [0.0 , 1.0]
    train, test = train[..., np.newaxis] / \
        255.0, test[..., np.newaxis] / 255.0
    '''
    Convert integer to float32 -> normalize to 0-1
    train = train.astype('float32')
    train = train/255.0
    '''
    return train, test


def define_model():

    model = tf.keras.Sequential()
    # Convolution layer with a (3,3) filter
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                                     kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.MaxPool2D((2, 2)))

    model.add(tf.keras.layers.Flatten())
    # Dense layer
    model.add(tf.keras.layers.Dense(100, activation='relu', kernel_initializer='he_uniform'))
    # Output layer with 10 nodes
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    # compile model
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def model_evaluation(x_data, y_data, n_folds=5):
    scores, histories = list(), list()
    # We will use k-fold cross validator to split data in train/test sets
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    # Enumerating the splits
    for train_ix, test_ix in kfold.split(x_data):
        # Defining the model we are going to use
        model = define_model()
        x_train, y_train, x_test, y_test = x_data[train_ix], y_data[train_ix], x_data[test_ix], y_data[test_ix]
        # Fitting the model
        history = model.fit(x_train, y_train, epochs=10, batch_size=32,
                            validation_data=(x_test, y_test), verbose=0)
        _, acc = model.evaluate(x_test, y_test, verbose=0)
        print('> %.3f' % (acc * 100.0))
        # stores scores
        scores.append(acc)
        histories.append(history)
    return scores, histories


# Plot diagnostic learning curves
def summarize_diagnostics(histories):
    for i in range(len(histories)):
        # plot loss
        plt.subplot(2, 1, 1)
        plt.title('Cross Entropy Loss')
        plt.plot(histories[i].history['loss'], color='blue', label='train')
        plt.plot(histories[i].history['val_loss'], color='orange', label='test')
        # plot accuracy
        plt.subplot(2, 1, 2)
        plt.title('Classification Accuracy')
        plt.plot(histories[i].history['accuracy'], color='blue', label='train')
        plt.plot(histories[i].history['val_accuracy'], color='orange', label='test')
    plt.show()

# Summarize model performance


def summarize_performance(scores):
        # print summary
    print('Accuracy: mean=%.3f std=%.3f, n=%d' %
          (np.mean(scores)*100, np.std(scores)*100, len(scores)))
    # box and whisker plots of results
    plt.boxplot(scores)
    plt.show()


def run_test_statistics():
    # Load the  dataset
    x_train, y_train, x_test, y_test = load_dataset()
    # Rescale the photo
    x_train, x_test = rescale_photo(x_train, x_test)
    # Evaluate model
    scores, histories = model_evaluation(x_train, y_train)
    # learning curves
    summarize_diagnostics(histories)
    # summarize estimated performance
    summarize_performance(scores)


def run_test():
    x_train, y_train, x_test, y_test = load_dataset()
    x_train, x_test = rescale_photo(x_train, x_test)
    model = define_model()
    model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)
    model.save('final_model.h5')


# To check the Classification Accuracy and print mean / std
# run_test_statistics()
# To train our model and save it
run_test()
