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
    _kwargs = kwargs.copy()
    parameters = kwargs.pop("parameters")
    model = kwargs.pop("model")

    for name, dataset in kwargs.items():
        type = name.split("_")[0]
        if type == "image":
            _image_dataset = dict(sorted(dataset.items()))
            ids = _image_dataset.keys()
        else:
            ids = dataset[parameters['ID']].values
    x, y = process_data(ids, _kwargs)
    last_layer = last_layer_normalized(model)
    last_layer = Dense(parameters['EMBEDDING_DIM'], activation='relu', name='Dense_Embedding')(last_layer)
    last_layer = BatchNormalization()(last_layer)
    model_headless = Model(inputs=model.input, outputs=last_layer)
    embedding = model_headless.predict(x)
    return {
        "embedding": embedding,
        "nodes": ids,
        "y": y,
        "name": name,
        "type": type,
        "parameters": parameters
    }