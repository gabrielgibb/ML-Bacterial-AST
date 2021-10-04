import tensorflow as tf
from tensorflow import keras
from functools import partial

def model(X_DIM, Y_DIM):
    """This is the CNN model we used to achieve our best results for both kanamycin and trimethoprim (95.1% and 77.6% accuracy respectively)"""
    DefaultConv2D = partial(keras.layers.Conv2D,
        kernel_size=5, activation='relu', kernel_initializer='he_normal', padding="SAME")
    DefaultDense = partial(keras.layers.Dense,
        activation='relu', kernel_initializer='he_normal')
    num_classes=2

    model = keras.models.Sequential([   
        DefaultConv2D(filters=32, kernel_size=5, strides=1, 
        input_shape=[X_DIM, Y_DIM, 1]),
        DefaultConv2D(filters=32),                            
        keras.layers.MaxPooling2D(pool_size=2),  # divides each spatial dimension by 2
        DefaultConv2D(filters=64),
        DefaultConv2D(filters=64),
        keras.layers.MaxPooling2D(pool_size=2),
        DefaultConv2D(filters=128),
        DefaultConv2D(filters=128),
        keras.layers.MaxPooling2D(pool_size=2),
        DefaultConv2D(filters=256),
        DefaultConv2D(filters=256),
        keras.layers.MaxPooling2D(pool_size=2),
        keras.layers.Flatten(),
        keras.layers.BatchNormalization(),
        DefaultDense(64),
        keras.layers.Dropout(0.5),
        keras.layers.BatchNormalization(),
        DefaultDense(32),
        keras.layers.Dropout(0.5),
        keras.layers.BatchNormalization(),
        DefaultDense(8),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation='softmax')
        ])
    return model