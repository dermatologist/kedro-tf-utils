"""
 Copyright 2023 Bell Eapen

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
from keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    Flatten,
)
from keras.models import Model
import numpy as np
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
import tensorflow as tf
import logging
import click
import sys
from kedro_tf_utils.pipelines.fusion.nodes import last_layer_normalized
from kedro_tf_utils.pipelines.train.nodes import get_intersection_ids, process_data
logger = logging.getLogger(__name__)

def create_embedding(**kwargs):
    """_summary_

    Returns:
        _type_: _description_
    """
    parameters = kwargs["parameters"]
    model = kwargs["model"]
    # Get intersection of all IDs
    intersection_ids = get_intersection_ids(parameters, kwargs)
    x, y = process_data(intersection_ids, parameters, kwargs)
    model_headless = last_layer_normalized(model)
    model_headless = Dense(parameters['EMBEDDING_DIM'], activation='relu', name='Dense_Embedding')(model_headless)
    model_headless = Model(inputs=model_headless.input, outputs=model_headless.get_layer('Dense_Embedding').output)
    embedding = model_headless.predict(x)
    return embedding