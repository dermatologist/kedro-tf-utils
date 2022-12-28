"""
This is a boilerplate pipeline 'train'
generated using Kedro 0.18.4
"""
import numpy as np
from keras.optimizers import Adam

def train_multimodal(**kwargs):
    """
    Train multimodal model
    """
    parameters = kwargs.pop("parameters")
    model = kwargs.pop("model")
    x = []
    y = None
    for name, dataset in kwargs.items():
        type = name.split("_")[0]
        if parameters['ID'] in dataset.keys():
            dataset = dataset.sort_values(by=[parameters['ID']])
        if type == "image":
            _image_dataset = dict(sorted(dataset.items()))
            ids = _image_dataset.keys()
            # column is a function that returns image data
            imgs = np.array([np.array(_image_dataset[id]()) for id in ids])
            x.append(imgs)
        elif type == "tabular":
            if parameters['TARGET'] in dataset.keys():
                y = dataset.pop(parameters['TARGET'])
            dataset.drop(parameters['DROP'], axis=1, inplace=True)
            csv_features_dict = {name: np.array(value)
                                 for name, value in dataset.items()}
            x.append(csv_features_dict)
        elif type == "bert":
            reports = dataset.pop(parameters['REPORT_FIELD'])
            if parameters['TARGET'] in dataset.keys():
                y = dataset.pop(parameters['TARGET'])
            x.append(reports)
        elif type == "text":
            reports = dataset.pop(parameters['REPORT_FIELD'])
            if parameters['TARGET'] in dataset.keys():
                y = dataset.pop(parameters['TARGET'])
            x.append(reports)
        else:
            raise ValueError("Unknown dataset type")
    model.compile(loss='binary_crossentropy',
                            optimizer=Adam(), metrics=['accuracy'])

    hist = model.fit(
        x=x,
        y=y, batch_size=32, epochs=3, verbose=1,
        validation_data=None)
    return model
