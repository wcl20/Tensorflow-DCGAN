import cv2
import numpy as np
import os
from core.nn import DCGAN
from imutils import build_montages
from sklearn.utils import shuffle
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam



def main():

    print("[INFO] Loading MNIST dataset ...")
    (X_train, _), (X_test, _) = mnist.load_data()
    # Use all images as training data
    X_train = np.concatenate([X_train, X_test])
    # Add extra dimension Nx28x28x1
    X_train = np.expand_dims(X_train, axis=-1)
    # Rescale values to output of tanh [-1, 1]
    X_train = X_train.astype("float")
    X_train = (X_train - 127.5) / 127.5

    epochs = 50
    batch_size = 128

    print("[INFO] Building generator ...")
    generator = DCGAN.build_generator(100, 512, (7, 7, 64), channels=1)

    print("[INFO] Building discriminator ...")
    discriminator = DCGAN.build_discriminator(28, 28, 1)
    disc_optimizer = Adam(lr=0.0002, beta_1=0.5, decay=0.0002 / epochs)
    discriminator.compile(loss="binary_crossentropy", optimizer=disc_optimizer)

    print("[INFO] Building GAN ...")
    discriminator.trainable = False
    gan_input = Input(shape=(100, ))
    gan_output = discriminator(generator(gan_input))
    gan = Model(gan_input, gan_output)
    gan_optimizer = Adam(lr=0.0001, beta_1=0.5, decay=0.0002 / epochs)
    gan.compile(loss="binary_crossentropy", optimizer=gan_optimizer)

    os.makedirs("output", exist_ok=True)
    benchmark_noise = np.random.uniform(-1, 1, size=(256, 100))

    print("[INFO] Training ...")
    for epoch in range(epochs):
        print(f"[INFO] Epoch {epoch + 1} / {epochs} ...")
        batch_per_epoch = X_train.shape[0] // batch_size
        for i in range(batch_per_epoch):
            # Create real batch from training image
            real_batch = X_train[i * batch_size : (i+1) * batch_size]
            # Create fake batch from generator
            noise = np.random.uniform(-1, 1, size=(batch_size, 100))
            fake_batch = generator.predict(noise, verbose=0)
            # Combine images and shuffle
            X = np.concatenate((real_batch, fake_batch))
            y = np.array([1] * batch_size + [0] * batch_size)
            X, y = shuffle(X, y)
            # Train discriminator (Discriminate real and fake)
            disc_loss = discriminator.train_on_batch(X, y)
            # Train GAN.
            noise = np.random.uniform(-1, 1, size=(batch_size, 100))
            gan_loss = gan.train_on_batch(noise, np.array([1] * batch_size))

            if i == batch_per_epoch - 1:
                images = generator.predict(benchmark_noise)
                images = (images * 127.5 + 127.5).astype("uint8")
                images = np.repeat(images, 3, axis=-1)
                visualize = build_montages(images, (28, 28), (16, 16))[0]
                cv2.imwrite(os.path.sep.join(["output", f"{epoch}.png"]), visualize)



if __name__ == '__main__':
    main()
