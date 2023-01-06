"""
This is a boilerplate pipeline 'train'
generated using Kedro 0.18.4
"""
import re
import numpy as np
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
import tensorflow as tf
from keras import Model

import logging
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
            dataset = dataset[dataset[parameters['ID']].isin(intersection_ids)]
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

    ## https: // stackoverflow.com/questions/49079115/valueerror-negative-dimension-size-caused-by-subtracting-2-from-1-for-max-pool
    model.compile(loss='binary_crossentropy',
                            optimizer=Adam(), metrics=['accuracy'])


    evaluate = parameters.get('EVALUATE', False)
    if evaluate:
        logger.info("Evaluating model")
        score = model.evaluate(x, y, batch_size=parameters.get('BATCH_SIZE', 32), verbose=1)
        logger.info("Score: {}".format(score))
        return score
    else:
        logger.info("Training model")
        hist = model.fit(
            x=x,
            y=y,
            batch_size=parameters.get('BATCH_SIZE', 32),
            epochs=parameters.get('EPOCHS', 3),
            verbose='auto',
            callbacks=None,
            validation_split=0.0,
            validation_data=None,
            shuffle=False,
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
        b64 = parameters.get('BASE64', True)
        if b64:
            model = insert_first_layer(model, 'input_1', None, 'b64_input_bytes', position='after')
    # print(model.summary())
    # print(model.inputs)
    return model





def insert_first_layer(model, layer_name, insert_layer_factory,
                        insert_layer_name=None, position='after'):

    inputs = tf.keras.layers.Input(shape=(), dtype=tf.string, name='b64_input_bytes')
    x = tf.keras.layers.Lambda(preprocess_input, name='decode_image_bytes')(inputs)
    idx =0
    for l in model.layers:
        if l.name == layer_name:
           x = l(x)
        model.layers[idx] = x
        idx += 1
    idx = 0
    for input in model.inputs:
        print("Input:{} {}".format(idx, input.name))
        if input.name == layer_name:
            print("Found input")
            model.inputs[idx] = x
        idx += 1
    print(model.inputs)
    model = tf.keras.Model(inputs=model.inputs, outputs=model.outputs)
    return model


def preprocess_input(base64_input_bytes):
    def decode_bytes(img_bytes):
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(img, (224, 224))
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img

    base64_input_bytes = tf.reshape(base64_input_bytes, (-1,))
    return tf.map_fn(lambda img_bytes:
                     decode_bytes(img_bytes),
                     elems=base64_input_bytes,
                     fn_output_signature=tf.float32)


# Load png encoded image from string placeholder
def png_to_input_tensor(png_placeholder, width=224, height=224, color_channels=3):

	input_tensor = tf.reshape(png_placeholder, [])
	input_tensor = tf.image.decode_png(input_tensor, channels=color_channels)

	# Convert image to float and bring values in the range of 0-1
	input_tensor = tf.image.convert_image_dtype(input_tensor, dtype=tf.float32)

	# Reshape and add "batch" dimension (this expects a single image NOT in a list)
	input_tensor = tf.reshape(input_tensor, [height, width, color_channels])
	# input_tensor = tf.expand_dims(input_tensor, 0)

	return input_tensor
