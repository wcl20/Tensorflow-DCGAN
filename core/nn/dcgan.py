import numpy as np
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import Sequential

class DCGAN:

    @staticmethod
    def build_generator(input_dim, output_dim, input_shape, channels):

        channel_dim = -1
        model = Sequential()

        # Increase dimension of noise vector
        model.add(Dense(input_dim=input_dim, units=output_dim, activation="relu"))
        model.add(BatchNormalization())

        # Increase dimension to size of input shape
        model.add(Dense(np.prod(input_shape), activation="relu"))
        model.add(BatchNormalization())

        # Reshape noise vector to volume
        model.add(Reshape(input_shape))

        # Increase image size by 2 times
        model.add(Conv2DTranspose(32, (5, 5), strides=(2, 2), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channel_dim))

        # Increase image size by 2 times.
        # Final color channel 1 for grayscale, 3 for rgb
        model.add(Conv2DTranspose(channels, (5, 5), strides=(2, 2), padding="same"))
        model.add(Activation("tanh"))

        return model

    @staticmethod
    def build_discriminator(height, width, channels, alpha=0.2):

        model = Sequential()
        input_shape = (height, width, channels)

        model.add(Conv2D(32, (5, 5), strides=(2, 2), padding="same", input_shape=input_shape))
        model.add(LeakyReLU(alpha=alpha))

        model.add(Conv2D(64, (5, 5), strides=(2, 2), padding="same"))
        model.add(LeakyReLU(alpha=alpha))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=alpha))

        model.add(Dense(1, activation="sigmoid"))

        return model
