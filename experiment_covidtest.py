"""Reusable experiment functions extracted from Covidtest.ipynb."""

from __future__ import annotations


def view_random_image(target_dir, target_class):
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    import os
    import random

    target_folder = target_dir + target_class
    random_image = random.sample(os.listdir(target_folder), 1)  # nosec B311
    img = mpimg.imread(target_folder + "/" + random_image[0])
    plt.imshow(img)
    plt.title(target_class)
    plt.axis("off")
    print(f"Image shape: {img.shape}")
    return img


def build_model_cell_14():
    """Build the original architecture without training or accessing data."""
    import tensorflow as tf

    return tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                filters=10,
                kernel_size=(3, 3),
                activation="relu",
                input_shape=(224, 224, 3),
            ),
            tf.keras.layers.MaxPool2D(pool_size=2, padding="valid"),
            tf.keras.layers.Conv2D(10, 3, activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
