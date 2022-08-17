import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, BatchNormalization, Activation, Dropout
from keras import regularizers


def build_1d_cnn_model(n_points=256):
    layers = 0
    tf.keras.backend.clear_session()
    model = Sequential()
    model.add(tf.keras.Input(shape=(n_points, 1)))

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
    layers += 1

    # Conv1D is for pattern detection
    # there are 128 kinds of filters and kernel size (32) is convolution window
    # adds 128 (filters) to the next dimension
    # reduces samples by 32 but adds 1 --> 640 - 32 + 1 = 609
    # model1.add(Conv1D(128, activation = 'relu', kernel_size = (64)))
    # layers += 1

    model.add(Conv1D(64, activation='relu', kernel_size=(32)))
    layers += 1

    # do 64 filters, 16 window kernel
    # model1.add(Conv1D(32, activation = 'relu', kernel_size = (16)))
    # layers += 1

    # so on...
    model.add(Conv1D(16, activation='relu', kernel_size=(8)))
    layers += 1
    # model1.add(Conv1D(8, activation = 'relu', kernel_size = (4)))
    # layers += 1

    model.add(Conv1D(4, activation='relu', kernel_size=(2)))
    layers += 1

    # model1.add(Conv1DTranspose(32, activation = 'relu', kernel_size = (16)))
    # layers += 1

    # model1.add(Conv1DTranspose(n_points, activation = 'relu', kernel_size = (int(n_points/2))))
    # layers += 1

    # Dense Layers == Fully connected layers
    # 32 units, 573 features
    model.add(Dense(3, activation='relu',
                    kernel_regularizer=regularizers.l1_l2(l1=0, l2=0.1)))
    layers += 1

    # 16 units, 573 features
    # model1.add(Dense(16, activation = 'relu', kernel_regularizer=regularizers.l1_l2(l1 = 0, l2=0.1)))
    # layers += 1
    # Flattens out the previous layer
    # 573 * 16 = 9168
    model.add(Flatten())

    # Dropout. With rate 0.25. Randomly sets a value to zero to prevent overfitting
    model.add(Dropout(0.25))
    layers += 1

    # Return to previous number of data points
    model.add(Dense(n_points))
    # model1.add(Dense(2, activation='relu'))
    layers += 1

    # loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    loss_fn = tf.keras.losses.MeanSquaredError()
    model.compile(loss=loss_fn, optimizer='Adam', metrics=['mean_absolute_error', 'mse', 'accuracy'])
    model.summary()
    return model
