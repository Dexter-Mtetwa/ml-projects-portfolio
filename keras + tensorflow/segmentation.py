from tensorflow import keras
from tensorflow.keras import layers

def get_model(num_class, img_size):
    inputs = keras.Input(shape=(num_class + (3,))) # 3 is the channel, for RGB so shape becomes something like (180, 180, 3)
    x = layers.Rescaling(1./255) (inputs)

    # we're not gonna use MaxPooling2D here, so we'll make use of 'strides=2' implementation in our Conv layers + padding='same' to avoid border padding
    x = layers.Conv2D(64, 3, strides=2, activation='relu', padding='same') (x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same') (x)
    x = layers.Conv2D(128, 3, strides=2, activation='relu', padding='same') (x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same') (x)
    x = layers.Conv2D(256, 3, strides=2, activation='relu', padding='same') (x)
    x = layers.Conv2D(256, 3, activation='relu', padding='same') (x)

    # since the target is an image with same shape as input image, i think that's why we transpose so we maintain the original shape of the image when inputted
    x = layers.Conv2DTranspose(256, 3, activation='relu', padding='same') (x)
    x = layers.Conv2DTranspose(256, 3, strides=2, activation='relu', padding='same') (x)
    x = layers.Conv2DTranspose(128, 3, activation='relu', padding='same') (x)
    x = layers.Conv2DTranspose(128, 3, strides=2, activation='relu', padding='same') (x)
    x = layers.Conv2DTranspose(64, 3, activation='relu', padding='same') (x)
    x = layers.Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same') (x)

    # output layer that classifies each output pixel into 1 of the 3 categories, so it's performing a 3-way softmax on each of the pixels instead of on the entire array / image
    # remember the targerts are not cat or dog, they're an image of the same size as the input image but with each pixel containing either 0 - fg, 1 - bg, 2 - contour
    outputs = layers.Conv2D(num_class, 3, activation='softmax', padding='same') (x)
    model = keras.Model(inputs, outputs)
    return model