"""
This is a boilerplate pipeline 'train'
generated using Kedro 0.18.4
"""
import numpy as np
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
import tensorflow as tf
import logging
def train_multimodal(**kwargs):
    """
    Train multimodal model
    """
    parameters = kwargs.pop("parameters")
    model = kwargs.pop("model")
    x = []
    y = None

    ## Get intersection of all IDs #############################################
    members = {}
    for name, dataset in kwargs.items():
        type = name.split("_")[0]
        if type == "image":
            _image_dataset = dict(sorted(dataset.items()))
            ids = _image_dataset.keys()
            members[name] = ids
        else:
            members[name] = dataset[parameters['ID']].values
    intersection_ids = set.intersection(*map(set, members.values()))
    logging.info("Intersection of IDs: {}".format(len(intersection_ids)))
    ## Get intersection of all IDs #############################################

    for name, dataset in kwargs.items():
        type = name.split("_")[0]
        try:
            if parameters['ID'] in dataset.keys():
                dataset = dataset.sort_values(by=[parameters['ID']])
        except:
            pass
        # Get data from processed dataset and Y from original csv dataset (below)
        if type == "processed":
            dataset = dataset[dataset[parameters['ID']].isin(intersection_ids)]
            x.append(dataset)
            logging.info("Text Dataset shape: {}".format(dataset.shape))  # (4,140)
        elif type == "image":
            _image_dataset = dict(sorted(dataset.items()))
            ids = _image_dataset.keys()
            # Filter out IDs that are not in intersection
            for id in list(ids):  # REF: https://stackoverflow.com/questions/11941817/how-to-avoid-runtimeerror-dictionary-changed-size-during-iteration-error
                if id not in intersection_ids:
                    del _image_dataset[id]
            # column is a function that returns image data
            imgs = [_image_dataset[id]().squeeze() for id in ids]
            imgs = np.array(imgs)
            logging.info("Image dataset shape: {}".format(imgs.shape))  # (4, 224, 224, 3)
            x.append(imgs)
        elif type == "tabular":
            dataset = dataset[dataset[parameters['ID']].isin(intersection_ids)]
            if parameters['TARGET'] in dataset.keys():
                y = dataset.pop(parameters['TARGET'])
            dataset.drop(parameters['DROP'], axis=1, inplace=True)
            csv_features_dict = {name: np.array(value)
                                 for name, value in dataset.items()}
            logging.info("Tabular dataset")  # List
            x.append(csv_features_dict)
        elif type == "bert":
            dataset = dataset[dataset[parameters['ID']].isin(intersection_ids)]
            reports = dataset.pop(parameters['REPORT_FIELD'])
            if parameters['TARGET'] in dataset.keys():
                y = dataset.pop(parameters['TARGET'])
            logging.info("BERT dataset")
            x.append(reports)
        # Get data from processed dataset (above) and Y from original csv dataset here
        elif type == "text":
            dataset = dataset[dataset[parameters['ID']].isin(intersection_ids)]
            if parameters['TARGET'] in dataset.keys():
                y = dataset.pop(parameters['TARGET'])
        else:
            raise ValueError("Unknown dataset type")

    ## https: // stackoverflow.com/questions/49079115/valueerror-negative-dimension-size-caused-by-subtracting-2-from-1-for-max-pool
    model.compile(loss='binary_crossentropy',
                            optimizer=Adam(), metrics=['accuracy'])

    hist = model.fit(
        x=x,
        y=y, batch_size=32, epochs=3, verbose=1,
        validation_data=None)
    return model
