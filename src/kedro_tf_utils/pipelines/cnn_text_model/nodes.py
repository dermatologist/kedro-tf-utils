"""
This is a boilerplate pipeline 'model'
generated using Kedro 0.18.1
"""
import tensorflow as tf
from keras.layers import Activation, Input, Dense, Flatten, Dropout, Embedding, BatchNormalization
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import concatenate
from keras import regularizers
from keras.models import Model
from keras.optimizers import Adadelta, Adam, SGD
from keras.applications.densenet import DenseNet121

import numpy as np
##### https://github.com/faikaydin/medical-multimodal-with-transfer-learning/blob/master/cnn_model.py
def create_channel(x, filter_size, feature_map):
    """
    Creates a layer working channel wise
    """
    x = Conv1D(feature_map, kernel_size=filter_size, activation='relu', strides=1,
               padding='same', kernel_regularizer=regularizers.l2(0.03))(x)
    x = MaxPooling1D(pool_size=2, strides=1, padding='valid')(x)
    x = Flatten()(x)
    return x


def create_cnn_model(embedding_layer=None, num_words=None,
              embedding_dim=None, filter_sizes=[3, 4, 5],
              feature_maps=[100, 100, 100], max_seq_length=100, dropout_rate=None, multi=False):

    if len(filter_sizes) != len(feature_maps):
        raise Exception(
            'Please define `filter_sizes` and `feature_maps` with the same length.')
    if not embedding_layer and (not num_words or not embedding_dim):
        raise Exception(
            'Please define `num_words` and `embedding_dim` if you not use a pre-trained embedding')

    if embedding_layer is None:
        embedding_layer = Embedding(input_dim=num_words, output_dim=embedding_dim,
                                    input_length=max_seq_length,
                                    weights=None,
                                    trainable=True
                                    )

    channels = []
    x_in = Input(shape=(max_seq_length,), dtype='int32')
    emb_layer = embedding_layer(x_in)
    if dropout_rate:
        emb_layer = Dropout(dropout_rate)(emb_layer)
    for ix in range(len(filter_sizes)):
        x = create_channel(emb_layer, filter_sizes[ix], feature_maps[ix])
        channels.append(x)

    # Concatenate all channels
    x = concatenate(channels)
    concat = concatenate(channels)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    x = Activation('relu')(x)
    x = Dense(1, activation='sigmoid')(x)
    if multi:
        return concat
    return Model(inputs=x_in, outputs=x)

##############

def train_cnn_model(emb_layer, X_train_text, y_train, X_test_text, y_test, parameters):
    model = create_cnn_model(
        embedding_layer=emb_layer,
        num_words=parameters['MAX_NUM_WORDS'],
        embedding_dim=parameters['EMBEDDING_DIM'],
        filter_sizes=parameters['FILTER_SIZES'],
        feature_maps=parameters['FEATURE_MAPS'],
        max_seq_length=parameters['MAX_SEQ_LENGTH'],
        dropout_rate=parameters['DROPOUT_RATE']
    )

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adadelta(clipvalue=3),
        metrics=['accuracy']
    )

    history = model.fit(
                    np.array(X_train_text), y_train,
                    epochs=parameters['EPOCHS'],
                    validation_data=(np.array(X_test_text), y_test),
                    verbose=0
                    )

    return model, history

def text_image_model_fusion(text_last_layer, img_last_layer, parameters):
    fusion = concatenate([text_last_layer, img_last_layer])
    x = BatchNormalization()(fusion)
    x = Dense(512, activation='relu')(x)
    x = Dropout(.3)(x)
    x = BatchNormalization()(x)
    out = Dense(1, activation='softmax')(x)
    x_in = Input(shape=(parameters['MAX_SEQ_LENGTH'],), dtype='int32')
    image_input = Input(shape=(224, 224, 3))
    multi_model = Model([x_in, image_input], out)
    return multi_model


# def text_tabular_model_fusion(text_last_layer, tabular_last_layer, parameters):
#     fusion = concatenate([text_last_layer, tabular_last_layer])
#     x = BatchNormalization()(fusion)
#     x = Dense(512, activation='relu')(x)
#     x = Dropout(.3)(x)
#     x = BatchNormalization()(x)
#     out = Dense(1, activation='softmax')(x)
#     x_in = Input(shape=(parameters['MAX_SEQ_LENGTH'],), dtype='int32')
#     tabular_input = Input(shape=(parameters['DIM_OF_LAST_LAYER_FROM_TABULAR'],), dtype='int32')
#     multi_model = Model([x_in, tabular_input], out)
#     return multi_model


def last_layer_normalized(model):
    last_layer = model.layers[-2].output
    return BatchNormalization()(last_layer)


def early_fusion_mm(text_model, image_model, parameters):
    text_last_layer = last_layer_normalized(text_model)
    image_last_layer = last_layer_normalized(image_model)
    fusion = concatenate([text_last_layer, image_last_layer])
    x = BatchNormalization()(fusion)
    x = Dense(512, activation='relu')(x)
    x = Dropout(.3)(x)
    x = BatchNormalization()(x)
    out = Dense(1, activation='softmax')(x)
    multi_model = Model([text_model.inputs, image_model.inputs], out)
    return multi_model
