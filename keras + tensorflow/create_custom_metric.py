import tensorflow as tf
from tensorflow import keras

class RootMeanSquaredError(keras.metrics.Metric):
    def __init__(self, name='rmse', **kwargs):
        super().__init_(name=name, **kwargs)
        self.mse_sum = self.add_weight(name='mse_sum', initializer='zeros')
        self.total_samples = self.add_weight(name='total_samples', initializer='zeros', dtype='int32')

    def update_state(self, y_true, y_pred):
        y_true = tf.one_hot(y_true, depth=tf.shape(y_pred)[1])
        mse = tf.reduce_sum(tf.square(y_true - y_pred))
        self.mse_sum.assign_add(mse)
        num_samples = tf.shape(y_pred) [0]
        self.total_samples.assign_add(num_samples)

    def result(self):
        return tf.sqrt(self.mse_sum / tf.cast(self.num_samples, tf.float32))
    
    def reset_state(self):
        self.mse_sum.assign(0)
        self.total_samples.assign(0)


# Custom metrics can be used just like built-in ones. Let’s test-drive our own metric:
model = get_mnist_model()
model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy", RootMeanSquaredError()])
model.fit(train_images, train_labels,epochs=3, validation_data=(val_images, val_labels))
test_metrics = model.evaluate(test_images, test_labels)