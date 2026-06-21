# feature extraction on pretrained model
from tensorflow import keras
from tensorflow.keras import layers

# load the model - gonna use the vgg16 pretrained model
conv_base = keras.applications.vgg16.VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(180, 180, 3)
)

#say we have this data, of shape (180, 180, 3), labels are 10 classes
train_images, train_labels = [], []
val_images, val_labels = [], []



# Approach 1 - Fast, use pre-model for inference to produce features that we then feed to our classifier

# takes in input images and returns their feature equivalence
def obtain_features(images):
    preprocessed_input = keras.applications.vgg16.preprocess_input(images) # vgg16 requires input data to be processed a certain way
    features = conv_base.predict(preprocessed_input) # use model for inference to obtain features
    return features

train_features = obtain_features(train_images) # features are of shape (5, 5, 512)
val_features = obtain_features(val_images)

# okay good we have our features, funny enough at this point the job of the pre-model is complete
# now we create our own classifier

def get_output_classifer():
    inputs = keras.Input(shape=(5, 5, 512)) # this is also the shape of the last feature map in the pre-model
    x = layers.Flatten() (inputs)
    x = layers.Dropout(0.5) (x)
    outputs = layers.Dense(10, activation='softmax') (x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

clf_model = get_output_classifer()
history = clf_model.fit(train_features, train_labels, epochs=50, batc_size=128)
# this approach is very fast, but it's likely to overfit so to counter that we can try approach number 2



# Approach 2 - more expensive, chain together the pre-model + our classifier and train the data end to end on this 

# the reason i re-instantiated the model is coz the last one i'd set input_size which won't apply here
conv_base = keras.applications.vgg16.VGG16(
    weights='imagenet',
    include_top=False,
)

# freeze the weights of the pre-model 
conv_base.trainable = False

# define our data aug set
data_aug = keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.2)
])

def get_chained_model():
    inputs = keras.Input(shape=(180, 180, 3)) # this is also the shape of the last feature map in the pre-model
    x = data_aug(inputs)
    x = keras.applications.vgg16.VGG16.preprocess_input(x)
    x = conv_base(x)
    x = layers.Flatten() (inputs)
    x = layers.Dense(256) (x)
    x = layers.Dropout(0.5) (x)
    outputs = layers.Dense(10, activation='softmax') (x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model