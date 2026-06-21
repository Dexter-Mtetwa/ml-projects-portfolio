import tensorflow as tf
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
import math
import pandas as pd
import numpy as np

# lemme create our naive dense layer
class NaiveDenseLayer:
    def __init__(self, input_size, output_size, activation):
        self.activation = activation

        w_shape = (input_size, output_size)
        initial_w = tf.random.uniform(w_shape, minval=0, maxval=1e-1)
        self.W =  tf.Variable(initial_w)
    
        b_shape = (output_size,)
        initial_b = tf.zeros(b_shape)
        self.b = tf.Variable(initial_b)
    
    def __call__(self, inputs):
        return self.activation(tf.matmul(inputs, self.W) + self.b)
    
    @property
    def weights(self):
        return [self.W, self.b]
    

# and here our naive Sequential model
class NaiveSequentialModel:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x
    
    @property
    def weights(self):
        weights = []
        for layer in self.layers:
            weights += layer.weights
        return weights
        

# create the model
model = NaiveSequentialModel([
    NaiveDenseLayer(input_size=28*28, output_size=512, activation=tf.nn.relu),
    NaiveDenseLayer(input_size=512, output_size=10, activation=tf.nn.softmax)
])
assert len(model.weights) == 4


# now imma create the batch generator
class BatchGenerator:
    def __init__(self, x_data, labels, batch_size=128):
        self.x_data = x_data
        self.labels = labels
        self.batch_size = batch_size
        self.index = 0
        self.num_batches = math.ceil(len(x_data) / batch_size)

    def next(self):
        x_batch = self.x_data[self.index: self.index + self.batch_size]
        labels_batch = self.labels[self.index: self.index + self.batch_size]
        self.index += self.batch_size
        return x_batch, labels_batch


# one forward pass step, let's define it
def one_training_step(model, x_batch, y_batch):
    with tf.GradientTape() as tape:
        predictions = model(x_batch)
        loss_per_sample = tf.keras.losses.sparse_categorical_crossentropy(y_batch, predictions)
        average_loss = tf.reduce_mean(loss_per_sample)
    # calculate the gradients of the average_loss wrt to model weights
    gradients = tape.gradient(average_loss, model.weights)
    update_weights(gradients, model.weights)
    return average_loss


# # update the weights
# learning_rate = 1e-3

# def update_weights(gradients, weights):
#     for g, w in zip(gradients, weights):
#         # update the weights in the negative direction of the gradient
#         w.assign_sub(g * learning_rate)

optimizer = optimizers.SGD(learning_rate=1e-3)

def update_weights(gradients, weights):
    optimizer.apply_gradients(zip(gradients, weights))



# okay, the final beautiful step - the fit function
def fit(model, x_train, y_train, epochs, batch_size=128):
    for epoch in range(epochs):
        print('epoch: {}/{}'.format(epoch, epochs))
        # let's define the number of batches, so we know how many times to train
        batch_counter = BatchGenerator(x_train, y_train, batch_size)
        for batch in range(batch_counter.num_batches):
            x_batch, label_batch = batch_counter.next()
            loss = one_training_step(model, x_batch, label_batch)
            if batch % 100 == 0:
                print('loss at batch {}: {}'.format(batch, loss))




# finally my broski-bro, let's do this
mnist_dataset = pd.read_csv('C:\\Users\\Gaming Dell G7\\Desktop\\code\\datasets\\mnist_data.csv')

# X, y split
images, labels = mnist_dataset.drop('label', axis=1), mnist_dataset['label']

# train, eval split
train_images, eval_images, train_labels, eval_labels = train_test_split(images, labels, test_size=.2, stratify=labels, random_state=42)

print('tain_images.shape: ', train_images.shape)
print('tain_labels.shape: ', train_labels.shape)
print('eval_images.shape: ', eval_images.shape)
print('eval_labels.shape: ', eval_labels.shape)

# convert dtype to 'float32' then normalize
train_images = train_images.astype('float32') / 255
eval_images = eval_images.astype('float32') / 255

# train the model
fit(model, train_images, train_labels, epochs=10, batch_size=128)

# eval 
preds = model(eval_images)
preds = preds.numpy()
preds_labels = np.argmax(preds, axis=1)
matches = preds_labels == eval_labels
print(f'model accuracy: {matches.mean():.2f}')
