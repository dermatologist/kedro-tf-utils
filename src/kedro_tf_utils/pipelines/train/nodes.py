"""
This is a boilerplate pipeline 'train'
generated using Kedro 0.18.4
"""
import numpy as np
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
import tensorflow as tf
import logging
import click
import sys

from kedro_tf_utils.extras.classes.fusion_model import ServingWrapperModel
logger = logging.getLogger(__name__)

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
            if type != "processed": # processed dataset is not a dataframe, so it doesn't have an ID column. ID will come from original csv dataset
                members[name] = dataset[parameters['ID']].values
    intersection_ids = set.intersection(*map(set, members.values()))
    logger.info("Intersection of IDs before image loading: {}".format(len(intersection_ids)))
    ## Get intersection of all IDs #############################################

    for name, dataset in kwargs.items():
        type = name.split("_")[0]
        try:
            if parameters['ID'] in dataset.keys():
                dataset = dataset.sort_values(by=[parameters['ID']])
        except:
            pass

        if type == "image":
            _image_dataset = dict(sorted(dataset.items()))
            ids = _image_dataset.keys()
            # Filter out IDs that are not in intersection
            for id in list(ids):  # REF: https://stackoverflow.com/questions/11941817/how-to-avoid-runtimeerror-dictionary-changed-size-during-iteration-error
                if id not in intersection_ids:
                    del _image_dataset[id]
            imgs = []
            for id in _image_dataset.keys():
                # column is a function that returns image data
                try:
                    img = _image_dataset[id]().squeeze()
                    img = tf.convert_to_tensor(img, dtype=tf.float32)
                    if img.shape[-1] == 1:  # If grayscale, convert to RGB
                        img = tf.image.grayscale_to_rgb(img)
                    imgs.append(img)
                except:
                    # image failed to load
                    logger.info("Image failed to load: {}".format(id))
                    intersection_ids.remove(id)
            # convert back to numpy array
            imgs = np.array(imgs)
            logger.info("Image dataset shape: {}".format(imgs.shape))  # (4, 224, 224, 3)
            x.append(imgs)
        # Get data from processed dataset and Y from original csv dataset (below)
        elif type == "processed":
            #dataset = dataset[dataset[parameters['ID']].isin(intersection_ids)]
            # res = dict((k, dataset[k]) for k in intersection_ids if k in dataset)
            dataset = [dataset[k] for k in intersection_ids if k in dataset]
            dataset = np.array(dataset)
            x.append(dataset)
            logger.info("Text Dataset shape: {}".format(dataset.shape))  # (4,140)
        elif type == "tabular":
            dataset = dataset[dataset[parameters['ID']].isin(intersection_ids)]
            if parameters['TARGET'] in dataset.keys():
                y = dataset.pop(parameters['TARGET'])
            dataset.drop(parameters['DROP'], axis=1, inplace=True)
            csv_features_dict = {name: np.array(value)
                                 for name, value in dataset.items()}
            logger.info("Tabular dataset")  # List
            x.append(csv_features_dict)
        elif type == "bert":
            dataset = dataset[dataset[parameters['ID']].isin(intersection_ids)]
            reports = dataset.pop(parameters['REPORT_FIELD'])
            if parameters['TARGET'] in dataset.keys():
                y = dataset.pop(parameters['TARGET'])
            logger.info("BERT dataset")
            x.append(reports)
        # Get data from processed dataset (above) and Y from original csv dataset here
        elif type == "text":
            dataset = dataset[dataset[parameters['ID']].isin(intersection_ids)]
            if parameters['TARGET'] in dataset.keys():
                y = dataset.pop(parameters['TARGET'])
        else:
            raise ValueError("Unknown dataset type")

    logger.info("Final intersection of IDs: {}".format(len(intersection_ids)))

    callbacks_path = parameters.get('CALLBACKS', False)
    callbacks = None
    if callbacks_path:
        logger.info("Saving model for serving")
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=5, restore_best_weights=True),
                     tf.keras.callbacks.ModelCheckpoint(callbacks_path, 'accuracy', save_best_only=False)]


    ## https: // stackoverflow.com/questions/49079115/valueerror-negative-dimension-size-caused-by-subtracting-2-from-1-for-max-pool
    model.compile(loss='binary_crossentropy',
                            optimizer=Adam(), metrics=['accuracy'])


    evaluate = parameters.get('EVALUATE', False)
    if evaluate:
        logger.info("Evaluating model")
        score = model.evaluate(x, y, batch_size=parameters.get('BATCH_SIZE', 32), verbose=1)
        click.secho("Score: {}".format(score), fg='green')
        sys.exit(0)
    else:
        logger.info("Training model")
        hist = model.fit(
            x=x,
            y=y,
            batch_size=parameters.get('BATCH_SIZE', 4),
            epochs=parameters.get('EPOCHS', 10),
            verbose='auto',
            callbacks=callbacks,
            validation_split=0.0,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            validation_batch_size=None,
            validation_freq=1,
            max_queue_size=10,
            workers=parameters.get('WORKERS', 1),
            use_multiprocessing=False
        )
    serving = parameters.get('SERVING', False)
    if serving:
        logger.info("Adding serving wrapper")
        _model = ServingWrapperModel(model)
        tf.saved_model.save(_model, serving, signatures=_model.get_signatures())
    return model
