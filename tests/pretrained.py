import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

import eereid as ee

g=ee.ghost()
g=g(ee.prepros.subsample(0.1))
g["triplet_count"]=1000

import tensorflow as tf
from tensorflow.keras.applications import InceptionV3, MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# Load the InceptionV3 model without the top layer (fully connected layer)
#base_model = InceptionV3(weights='imagenet', include_top=False)
base_model = MobileNetV2(weights='imagenet', include_top=False)

# Add a global average pooling layer
x = base_model.output
#x = GlobalAveragePooling2D()(x)

# Add a fully connected dense layer with, for example, 1024 units
x = Dense(1024, activation='relu')(x)

# Add a final dense layer for classification, for example with 10 classes
predictions = Dense(100, activation='linear')(x)

# Create the new model
model = Model(inputs=base_model.input, outputs=predictions)

## Optionally, you can freeze the base_model layers to prevent them from being trained
#for layer in base_model.layers:
#    layer.trainable = False

# Compile the model
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary of the model to check the changes
model.summary()


g(ee.models.custom_model(model))(ee.prepros.resize((32,32)))(ee.prepros.add_color())










acc=g.evaluate()
print(acc)

