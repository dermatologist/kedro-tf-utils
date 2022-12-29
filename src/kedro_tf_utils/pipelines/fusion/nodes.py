"""
This is a boilerplate pipeline 'fusion'
generated using Kedro 0.18.1
"""
from typing import Dict
from keras.layers import Dense, Flatten, Dropout, BatchNormalization, AveragePooling2D
from keras import models, layers
from keras.models import Model
from keras.layers import LSTM
import logging
# https://github.com/keras-team/keras/issues/7403
def last_layer_normalized(model):
    last_layer = model.layers[-2].output
    try:
        return BatchNormalization()(last_layer)
    except:
        # https://stackoverflow.com/questions/58607787/bert-embedding-layer-raises-type-error-unsupported-operand-types-for-non
        last_layer = LSTM(128, name="LSTM", dropout=0.2,
                          recurrent_dropout=0.2, return_sequences=False)(model.output)
        output = Dense(128, activation="softmax")(last_layer)
        model_lstm = models.Model(model.input, output)
        return model_lstm.output

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
def early_fusion_mm(**kwargs):
    """_summary_

    Args:
        parameters (Dict): _description_

    Returns:
        _type_: _description_
    """
    parameters = kwargs.pop("parameters")
    models_headless = []
    input_shapes = []
    for name, model in kwargs.items():
        models_headless.append(last_layer_normalized(model))
        input_shapes.append(model.input)
    # A `Concatenate` layer requires inputs with matching shapes except for the concatenation axis.
    # Received: input_shape=[(None, 6), (None, 128, 128), (None, 1024)]
    fusion = layers.Concatenate(name="fusion_head_1")(models_headless)
    x = BatchNormalization()(fusion)
    x = Dense(512, activation='relu')(x)
    x = Dropout(.3)(x)
    x = BatchNormalization()(x)
    out = Dense(1, activation='softmax', name="fusion_1")(x)
    multi_model = Model(input_shapes, out)
    logging.info("Multi model summary: input_shapes: {}, out: {}".format(input_shapes, out))
    return multi_model
