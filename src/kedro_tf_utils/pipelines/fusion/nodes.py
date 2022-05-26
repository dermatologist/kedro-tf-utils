"""
This is a boilerplate pipeline 'fusion'
generated using Kedro 0.18.1
"""
import tensorflow as tf
from keras.layers import Activation, Input, Dense, Flatten, Dropout, Embedding, BatchNormalization
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import concatenate
from keras import regularizers
from keras.models import Model
from tensorflow.keras.optimizers import Adadelta, Adam, SGD
from keras.applications.densenet import DenseNet121

def last_layer_normalized(model):
    last_layer = model.layers[-2].output
    return BatchNormalization()(last_layer)


def early_fusion_mm(text_model, image_model, parameters):
    text_last_layer = last_layer_normalized(text_model)
    image_last_layer = last_layer_normalized(image_model)
    fusion = concatenate([text_last_layer, image_last_layer])
    x = BatchNormalization()(fusion)
    x = Dense(512, activation='relu')(x)
    x = Dropout(.3)(x)
    x = BatchNormalization()(x)
    out = Dense(1, activation='softmax')(x)
    multi_model = Model([text_model.inputs, image_model.inputs], out)
    return multi_model
