"""
This is a boilerplate pipeline 'fusion'
generated using Kedro 0.18.1
"""
from typing import Dict, List
import tensorflow as tf
from keras.layers import Activation, Input, Dense, Flatten, Dropout, Embedding, BatchNormalization, AveragePooling2D
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import concatenate
from keras import regularizers
from keras.models import Model
from tensorflow.keras.optimizers import Adadelta, Adam, SGD
from keras.applications.densenet import DenseNet121

def last_layer_normalized(model):
    last_layer = model.layers[-2].output
    return BatchNormalization()(last_layer)

# https://github.com/artelab/Image-and-Text-fusion-for-UPMC-Food-101-using-BERT-and-CNNs/blob/main/stacking_early_fusion_UPMC_food101.ipynb
# !Not currently used
def add_dense_layers(model, parameters):
    model.add(AveragePooling2D(pool_size=(8, 8), name='AVG_Pooling'))
    model.add(Dropout(.4, name='Dropout_0.4'))
    model.add(Flatten(name='Flatten'))
    model.add(Dense(128, name='Dense_128'))

    # Keep model layers trainable
    for layer in model.layers:
        layer.trainable = True
    return model


# ! Parameters come first followed by the models. Note this when using this node in the pipeline
def early_fusion_mm(parameters: Dict, *args):
    """_summary_

    Args:
        parameters (Dict): _description_

    Returns:
        _type_: _description_
    """
    models_headless = []
    input_shapes = []
    for model in args:
        models_headless.append(last_layer_normalized(model))
        input_shapes.append(model.input)
    fusion = concatenate(models_headless)
    x = BatchNormalization()(fusion)
    x = Dense(512, activation='relu')(x)
    x = Dropout(.3)(x)
    x = BatchNormalization()(x)
    out = Dense(1, activation='softmax')(x)
    multi_model = Model(input_shapes, out)
    return multi_model
