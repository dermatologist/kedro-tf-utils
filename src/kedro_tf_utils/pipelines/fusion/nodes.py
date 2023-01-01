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
        logging.info("Last layer is not a dense layer")
        last_layer = LSTM(128, name="LSTM", dropout=0.2,
                          recurrent_dropout=0.2, return_sequences=False)(model.output)
        output = Dense(128, activation="softmax", name="DENSE_128")(last_layer)
        model_lstm = models.Model(model.input, output, name="LSTM_Model")
        return model_lstm.output

# ! Parameters come first followed by the models. Note this when using this node in the pipeline
def early_fusion_mm(**kwargs) -> Model:
    """_summary_

    Returns:
        Model: Fusion model
    """
    parameters = kwargs.pop("parameters") # ! Parameters come first followed by the models. Note this when using this node in the pipeline
    models_headless = []
    input_shapes = []
    if parameters['EARLAY_FUSION']=='early':
        logging.info("Early fusion")
        for name, model in kwargs.items():
            for layer in model.layers:
                layer.trainable = True
    else:
        logging.info("Late fusion")
        for name, model in kwargs.items():
            for layer in model.layers:
                layer.trainable = False
    for name, model in kwargs.items():
        logging.info("Adding Model: {}".format(name))
        models_headless.append(last_layer_normalized(model))
        input_shapes.append(model.input)
    fusion = layers.Concatenate(name="fusion_head_1")(models_headless)
    x = BatchNormalization()(fusion)

    if parameters['EARLAY_FUSION']=='early':
        x = Dense(256, activation='relu', name="DENSE_256_fusion")(x)
        x = Dropout(.2)(x)
        out = Dense(parameters['NCLASSES'], activation='softmax', name="fusion_1")(x)
    else: # Late fusion
        x = layers.Dense(256, activation='relu', name='Dense_256')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(128, activation='relu', name='Dense_128')(x)
        x = layers.Dropout(0.2)(x)
        out = layers.Dense(parameters['NCLASSES'], activation='softmax', name="class")(x)
    multi_model = Model(input_shapes, out, name="fusion_model")
    logging.info("Multi model summary: input_shapes: {}, out: {}".format(input_shapes, out))
    return multi_model

