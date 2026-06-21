"""
3 inputs
i. title of ticket - text
ii. text body of ticket - text
iii. any tags added by user - categorical

2 ouputs
i. priority of ticket - scalar 0 or 1 (sigmoid)
ii. department to handle ticket - num of departments (sofmax)
"""

from tensorflow import keras

vocab_size = 10000
num_of_tags = 100
num_of_departments = 4

title = keras.Input(shape=(vocab_size,), name='ticket_title')
text_body = keras.Input(shape=(vocab_size,), name='text_body')
tags = keras.Input(shape=(num_of_tags,), name='user_defined_tags')

features = keras.layers.Concatenate() ([title, text_body, tags])
features = keras.layers.Dense(64, activation='relu') (features)

priority = keras.layers.Dense(1, activation='sigmoid', name='priority') (features)
department = keras.layers.Dense(num_of_departments, activation='softmax', name='department') (features)

model = keras.Model(inputs=([title, text_body, tags]), outputs=([priority, department]))

print(model.summary())