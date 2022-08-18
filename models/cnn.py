import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, BatchNormalization, Activation, Dropout
from keras import regularizers


def build_1d_cnn_model(n_in=256, n_out=256):
    """
    Convenience function to build a 1D CNN model.
    :param n_in: number of input data points
    :type n_in: int
    :param n_out: number of output data points
    :type n_out: int
    :return: A keras sequential model
    :rtype: Sequential
    """
    tf.keras.backend.clear_session()
    model = Sequential()
    model.add(tf.keras.Input(shape=(n_in, 1)))

    # Batch normalization
    model.add(BatchNormalization(axis=-1,
                                 momentum=0.99,
                                 epsilon=0.001,
                                 center=True,
                                 scale=True,
                                 beta_initializer='zeros',
                                 gamma_initializer='ones',
                                 moving_mean_initializer='zeros',
                                 moving_variance_initializer='ones',
                                 beta_regularizer=None,
                                 gamma_regularizer=None,
                                 beta_constraint=None,
                                 gamma_constraint=None,
                                 # input_shape = (n_points, 1) # can be deprecated by input layer
                                 ))

    # gives batch normalization activation 'relu'
    model.add(Activation('relu'))

    # Conv1D is for pattern detection
    # there are 128 kinds of filters and kernel size (32) is convolution window
    # adds 128 (filters) to the next dimension
    # reduces samples by 32 but adds 1 --> 640 - 32 + 1 = 609
    # model1.add(Conv1D(128, activation = 'relu', kernel_size = (64)))

    model.add(Conv1D(filters=32, activation='relu', kernel_size=64))
    model.add(Conv1D(filters=32, activation='relu', kernel_size=32))
    model.add(Conv1D(filters=16, activation='relu', kernel_size=8))
    model.add(Dense(10, activation='relu',
                    kernel_regularizer=regularizers.l1_l2(l1=0, l2=0.1)))

    model.add(Flatten())

    # Dropout. With rate 0.25. Randomly sets a value to zero to prevent overfitting
    model.add(Dropout(0.25))

    model.add(Dense(n_out))

    # loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    loss_fn = tf.keras.losses.MeanSquaredError()
    model.compile(loss=loss_fn, optimizer='Adam', metrics=['mean_absolute_error', 'mse', 'accuracy'])
    model.summary()
    return model
