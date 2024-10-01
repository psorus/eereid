import eereid as ee
import numpy as np
from plt import *
#import tensorflow as tf
#from tensorflow import keras


g=ee.ghost(ee.datasets.mnist(),ee.prepros.subsample(0.1),triplet_count=100)

acc=g.evaluate()


#def extract_activation_heatmap(image, model):
#    image=np.expand_dims(image,0)
#    def find_last_conv_layer_name(model):
#        for layer in reversed(model.layers):
#            if len(layer.output_shape) == 4:
#                return layer.name
#        raise ValueError("Could not find the last convolutional layer.")
#    # create a model that maps the input image to the activations
#    # of the last convolutional layer
#    activation_model = keras.models.Model(
#        model.inputs, model.get_layer(find_last_conv_layer_name(model)).output
#    )
#    # activation_model.summary()
#    # Get the feature maps for the input image
#    conv_layer_output = activation_model(image)
#
#    # Square the feature maps and sum across channels
#    # We want to sum across the channels (axis=-1)
#    activation_map = tf.reduce_sum(tf.square(conv_layer_output), axis=-1)
#
#    # Normalize the activation map between 0 and 1 for visualization
#    activation_map = tf.maximum(activation_map, 0)  # Ensure non-negative values
#    heatmap = activation_map[0] 
#    heatmap = heatmap / tf.math.reduce_max(heatmap)  # Normalize

#    return heatmap.numpy()




#img=g.gx[0]

g.plot_activation_heatmap(0)

#htm=extract_activation_heatmap(img,g.model.submodel)
#plt.imshow(img,cmap="gray")
#plt.imshow(htm,alpha=0.4,cmap="hot", extent=[0,len(img),0,len(img[0])])
plt.show()








