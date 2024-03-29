"""
This is a boilerplate pipeline 'train_mm_simple'
generated using Kedro 0.18.4
"""
"""
This is a boilerplate pipeline 'train'
generated using Kedro 0.18.1
"""




from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adadelta, Adam, SGD
from keras_preprocessing.image import img_to_array
import numpy as np
import tensorflow as tf
import keras.layers as layers
def train_multimodal(text_dataset, image_dataset, multi_modal, parameters):
    """
    Train multimodal model
    """

    # sort by key that is the same as parameters['ID']
    _image_dataset = dict(sorted(image_dataset.items()))
    _text_dataset = text_dataset.sort_values(by=[parameters['ID']])

    y = _text_dataset.pop(parameters['TARGET'])
    _text_dataset.drop(parameters['DROP'], axis=1, inplace=True)

    ids = _image_dataset.keys()
    # column is a function that returns image data
    imgs = np.array([np.array(_image_dataset[id]()) for id in ids])
    csv_features_dict = {name: np.array(value)
                         for name, value in _text_dataset.items()}

    multi_modal.compile(loss='binary_crossentropy',
                        optimizer=Adam(), metrics=['accuracy'])

    hist = multi_modal.fit(
        x=[csv_features_dict, imgs],
        y=y, batch_size=32, epochs=3, verbose=1,
        validation_data=None)
    return multi_modal


def train_multimodal_bert(text_dataset, bert_dataset, multi_modal, parameters):
    """
    Train multimodal model
    """

    # sort by key that is the same as parameters['ID']
    _text_dataset = text_dataset.sort_values(by=[parameters['ID']])

    y = _text_dataset.pop(parameters['TARGET'])
    _text_dataset.drop(parameters['DROP'], axis=1, inplace=True)
    csv_features_dict = {name: np.array(value)
                         for name, value in _text_dataset.items()}
    reports = bert_dataset.pop(parameters['REPORT_FIELD'])


    multi_modal.compile(loss='binary_crossentropy',
                        optimizer=Adam(), metrics=['accuracy'])

    hist = multi_modal.fit(
        x=[csv_features_dict, reports],
        y=y, batch_size=32, epochs=3, verbose=1,
        validation_data=None)
    return multi_modal
