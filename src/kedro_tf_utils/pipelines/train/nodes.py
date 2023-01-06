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
            model = insert_first_layer(model, 'input_1', dropout_layer_factory, 'b64_input_bytes', position='after')
    # print(model.summary())
    # print(model.inputs)
    return model


def dropout_layer_factory():
    inputs = tf.keras.layers.Input(shape=(), dtype=tf.string, name='b64_input_bytes')
    return tf.keras.layers.Lambda(preprocess_input, name='decode_image_bytes')(inputs)


def insert_first_layer(model, layer_name, insert_layer_factory,
                        insert_layer_name=None, position='after'):


    to_modify = None
    for layer in model.layers:
        if layer.name == layer_name:
            to_modify = layer
            for node in layer._outbound_nodes:
                layer_name = node.outbound_layer.name
                print("Outbound layer name: {}".format(layer_name))
            break
    else:
        raise ValueError('Layer not found')



    # Create the new layer
    new_layer = insert_layer_factory()
    # new_layer = to_modify(new_layer)
    print("New layer: {}".format(new_layer))

    for input in model.inputs:
        print("Input: {}".format(input.name))


    return model

# https://stackoverflow.com/questions/49492255/how-to-replace-or-insert-intermediate-layer-in-keras-model
def insert_layer_nonseq(model, layer_regex, insert_layer_factory,
                        insert_layer_name=None, position='after'):

    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

    # Set the input layers of each layer
    for layer in model.layers:
        # print(layer.name)
        for node in layer._outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update(
                    {layer_name: [layer.name]})
            else:
                network_dict['input_layers_of'][layer_name].append(layer.name)

    # Set the output tensor of the input layer
    network_dict['new_output_tensor_of'].update(
        {model.layers[0].name: model.inputs[0].name})



    # Iterate over all layers after the input
    model_outputs = []
    for layer in model.layers[1:]:

        # Determine input tensors
        layer_input = [network_dict['new_output_tensor_of'][layer_aux]
                       for layer_aux in network_dict['input_layers_of'][layer.name]]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        # Insert layer if name matches the regular expression
        if re.match(layer_regex, layer.name):
            if position == 'replace':
                x = layer_input
            elif position == 'after':
                x = layer(layer_input)
            elif position == 'before':
                pass
            else:
                raise ValueError('position must be: before, after or replace')

            new_layer = insert_layer_factory()
            if insert_layer_name:
                new_layer.name = insert_layer_name
            else:
                new_layer.name = '{}_{}'.format(layer.name,
                                                new_layer.name)
            x = new_layer(x)
            print('New layer: {} Old layer: {} Type: {}'.format(new_layer.name,
                                                                layer.name, position))
            if position == 'before':
                x = layer(x)
        else:
            x = layer(layer_input)

        # Set new output tensor (the original one, or the one of the inserted
        # layer)
        network_dict['new_output_tensor_of'].update({layer.name: x})

        # Save tensor in output list if it is output in initial model
        if layer_name in model.output_names:
            model_outputs.append(x)

    return Model(inputs=model.inputs, outputs=model_outputs)

 # https://github.com/tensorflow/serving/issues/1869?utm_source=pocket_saves
def preprocess_input(base64_input_bytes, parameters={}):
    def decode_bytes(img_bytes):
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(img, parameters.get('MODEL_INPUT_SHAPE', (224, 224)))
        img = tf.image.convert_image_dtype(img, parameters.get('MODEL_INPUT_DTYPE', tf.string))
        return img

    base64_input_bytes = tf.reshape(base64_input_bytes, (-1,))
    return tf.map_fn(lambda img_bytes:
                     decode_bytes(img_bytes),
                     elems=base64_input_bytes,
                     fn_output_signature=parameters.get('MODEL_INPUT_DTYPE', tf.string))
