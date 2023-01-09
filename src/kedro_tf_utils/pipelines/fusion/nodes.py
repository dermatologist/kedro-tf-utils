"""
This is a boilerplate pipeline 'fusion'
generated using Kedro 0.18.1
"""
import logging
from typing import Dict

from keras import layers, models
from keras.layers import (
    LSTM,
    AveragePooling2D,
    BatchNormalization,
    Dense,
    Dropout,
    Flatten,
)
from keras.models import Model
from kedro_tf_image.extras.classes.tf_image_classifier_model import ImageClassifierModel


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
def fusion(**kwargs) -> Model:
    """_summary_

    Returns:
        Model: Fusion model
    """
    parameters = kwargs.pop("parameters") # ! Parameters come first followed by the models. Note this when using this node in the pipeline
    models_headless = []
    input_shapes = []

    if parameters['EARLY_FUSION']: #True
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

    if parameters['EARLY_FUSION']:
        x = Dense(256, activation='relu', name="DENSE_256_fusion")(x)
        x = Dropout(.2)(x)
        out = Dense(parameters['NCLASSES'], activation='softmax', name="fusion_1")(x)
    else: # Late fusion
        x = Dense(256, activation='relu', name='Dense_256')(x)
        x = Dropout(0.2)(x)
        x = Dense(128, activation='relu', name='Dense_128')(x)
        x = Dropout(0.2)(x)
        out = Dense(parameters['NCLASSES'], activation='softmax', name="class")(x)

    multi_model = Model(input_shapes, out, name="fusion_model")
    logging.info("Multi model summary: input_shapes: {}, out: {}".format(input_shapes, out))
    visualize = parameters.get("VISUALIZE", False)
    if visualize:
        logging.info("Visualizing model")
        from keras.utils import plot_model
        plot_model(multi_model, to_file=parameters["VISUALIZE"], show_shapes=True, show_layer_names=True)
    return multi_model

