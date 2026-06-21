"""
In the __init__() method, define the layers the model will use
In the call() method, perform forwards pass
"""
from tensorflow import keras
from tensorflow.keras import layers

class CustomerTicketModel(keras.Model):
    def __init__(self, num_departments):
        super().__init__(self)
        self.concat_layer = layers.Concatenate()
        self.mixing_layer = layers.Dense(64, activation='relu')
        self.priority_scorer = layers.Dense(1, activation='sigmoid')
        self.department_classifier = layers.Dense(num_departments, activation='softmax')

    def call(self, inputs):
        title = inputs['title']
        text_body = inputs['text_body']
        tags = inputs['tags']

        features = self.concat_layer([title, text_body, tags])
        features = self.mixing_layer(features)
        priority = self.priority_scorer(features)
        department = self.department_classifier(features)

        return priority, department
    
model = CustomerTicketModel(num_departments=4)
priority, department = model({'title': title_data, 'text_body': text_body_data, 'tags': tags_data})

# difference between Model subclassing and Layer subclassing is, this one has access to the .fit() .compile() .evaluate() .predict() methods, plus you can save it to disk
