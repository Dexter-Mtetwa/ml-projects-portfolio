# Training a deep net from scratch using little data
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# generate my data, train images = (10000, 28, 28, 3), labels = (10000) either 0 or 1
images = tf.random.normal(shape=(10000, 28, 28, 3), seed=42)
labels_zeros = tf.zeros(shape=[5000])
labels_ones = tf.ones(shape=[5000])
labels = np.concat([labels_zeros, labels_ones])
np.random.shuffle(labels)

# split into train, val and test, train -> 5000, val -> 2000, test -> 3000
train_images, train_labels = images[:5000], labels[:5000]
val_images, val_labels = images[5000:7000], labels[5000:7000]
test_images, test_labels = images[7000:], labels[7000:]

print('train shape: ', train_images.shape)
print('val shape: ', val_images.shape)
print('test shape: ', test_images.shape)


# let's build our models
"""
# baseline model that hopefully overfits
def get_baseline_model():
    inputs = keras.Input(shape=(28, 28, 3))
    x = layers.Conv2D(filters=32, kernel_size=3, activation='relu') (inputs)
    x = layers.MaxPooling2D(pool_size=2) (x)
    x = layers.Conv2D(filters=64, kernel_size=3, activation='relu') (x)
    x = layers.MaxPooling2D(pool_size=2) (x)
    x = layers.Conv2D(filters=128, kernel_size=3, activation='relu') (x)
    x = layers.Flatten() (x) # Dense layer takes in 1D tensors, hence Flatten
    outputs = layers.Dense(1, activation='sigmoid') (x)
    model = keras.Model(inputs, outputs)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

# let's train the model
# baseline_model = get_baseline_model()
# print(baseline_model.summary())

callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='practice_model_w_aug.keras',
        monitor='val_loss',
        save_best_only=True,
    )
]


baseline_history = baseline_model.fit(train_images, train_labels, epochs=20, batch_size=128, callbacks=callbacks_list, validation_data=[val_images, val_labels])

# let's plot
val_acc, val_loss = baseline_history.history['val_accuracy'], baseline_history.history['val_loss']
train_acc, train_loss = baseline_history.history['accuracy'], baseline_history.history['loss']
epochs = range(1, len(val_acc) + 1)
plt.plot(epochs, train_acc, 'b', label='train accuracy')
plt.plot(epochs, val_acc, 'r', label='validation accuracy')
plt.title('Train and Validation accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()
"""

# evaluate on test set
saved_baseline_model = keras.models.load_model('practice_model.keras')
test_loss, test_acc = saved_baseline_model.evaluate(test_images, test_labels)
print(f'test_acc: {test_acc}, test_loss: {test_loss}')




"""
# okay so model #2 - data_aug + dropout let's go
data_aug = keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.2)
])

# essentially baseline + data_aug + dropout
def get_regularized_model():
    inputs = keras.Input(shape=(28, 28, 3))
    x = data_aug(inputs)
    x = layers.Conv2D(filters=32, kernel_size=3, activation='relu') (x)
    x = layers.MaxPooling2D(pool_size=2) (x)
    x = layers.Conv2D(filters=64, kernel_size=3, activation='relu') (x)
    x = layers.MaxPooling2D(pool_size=2) (x)
    x = layers.Conv2D(filters=128, kernel_size=3, activation='relu') (x)
    x = layers.Flatten() (x) # Dense layer takes in 1D tensors, hence Flatten
    x = layers.Dropout(0.5) (x)
    outputs = layers.Dense(1, activation='sigmoid') (x)
    model = keras.Model(inputs, outputs)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

regularized_model = get_regularized_model()
regularized_history = regularized_model.fit(train_images, train_labels, epochs=20, batch_size=128, callbacks=callbacks_list, validation_data=[val_images, val_labels])

# let's plot
val_acc, val_loss = regularized_history.history['val_accuracy'], regularized_history.history['val_loss']
train_acc, train_loss = regularized_history.history['accuracy'], regularized_history.history['loss']
epochs = range(1, len(val_acc) + 1)
plt.plot(epochs, train_acc, 'b', label='train accuracy')
plt.plot(epochs, val_acc, 'r', label='validation accuracy')
plt.title('Train and Validation accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()
"""

# evaluate on test set
saved_regularized_model = keras.models.load_model('practice_model_w_aug.keras')
test_loss, test_acc = saved_regularized_model.evaluate(test_images, test_labels)
print(f'test_acc: {test_acc}, test_loss: {test_loss}')