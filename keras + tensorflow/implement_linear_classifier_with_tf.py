# we wanna implement a linear classifier using pure Tensorflow
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

### 1. Data - generate data [X & y]
# X data
num_of_samples = 1000

class_1 = np.random.multivariate_normal(
    mean=[0, 3],
    cov=[[1, 0.5], [0.5, 1]],
    size=num_of_samples
)
# print('class_1.shape: ', class_1.shape)

# class 2 - another 1000 instances
class_2 = np.random.multivariate_normal(
    mean=[3, 0],
    cov=[[1, 0.5], [0.5, 1]],
    size=num_of_samples
)
# print('class_2.shape: ', class_2.shape)

# stack them together to create one input dataset
inputs = np.vstack((class_1, class_2), dtype=(np.float32))
print('inputs.shape: ', inputs.shape)


# y data - has to be of shape (X_data.shape[0], 1)
target_data_shape = (num_of_samples, 1)
# we're gonna vertically stack two (1000, 1) arrays, one with all zeros and the other with all ones
target_array_1 = np.zeros(target_data_shape, dtype='float32')
target_array_2 = np.ones(target_data_shape, dtype='float32')
targets = np.vstack((target_array_1, target_array_2))
print('targets.shape: ', targets.shape)


# let's see what our data looks like
# think of our data as (x,y) coordinates on a 2d plane
# plt.scatter(inputs[:, 0], inputs[:, 1], c=targets)
# plt.show()


### 2. Building the Model

# weights
input_dim = 2
output_dim = 3

W = tf.Variable(initial_value=tf.random.uniform(shape=(input_dim, output_dim), minval=0, maxval=1))
b = tf.Variable(initial_value=tf.zeros(shape=(output_dim,)))

print('W.shape: ', W.shape)
print('b.shape', b.shape)

# forward pass function
def model(inputs):
    return tf.matmul(inputs, W) + b

# loss function
def square_loss(targets, predictions):
    loss_per_sample = tf.square(targets - predictions)
    return tf.reduce_mean(loss_per_sample)

# training function
learning_rate = .1
def one_training_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = square_loss(targets, predictions)
    gradient_loss_wrt_W, gradient_loss_wrt_b = tape.gradient(loss, [W, b])
    W.assign_sub(gradient_loss_wrt_W * learning_rate)
    b.assign_sub(gradient_loss_wrt_b * learning_rate)
    return loss


### 3. Tain the model
# here we're doing batch training (not mini-batch), meaning the entire batch is supplied at each training step
for step in range(100):
    loss = one_training_step(inputs, targets)
    print(f'{step+1}/{40}')
    print('loss: {}'.format(loss))