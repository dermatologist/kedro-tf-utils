import tensorflow as tf
from keras.layers import Input, Dense, Conv1D, Dropout, Flatten, Conv2D, MaxPooling1D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, Embedding
from keras.models import Model
from keras.applications.densenet import DenseNet121
from deeptables.models.layers import dt_custom_objects
from keras.layers import concatenate
import pickle

model = tf.keras.models.load_model('model/tabular_model.h5', dt_custom_objects)
last_layer = model.layers[-2].output
text_last_layer = BatchNormalization()(last_layer)


image_input = Input(shape=(224, 224, 3))
base_model = DenseNet121(include_top=True, input_tensor=image_input,
                         input_shape=None, pooling=None, classes=1000)
last_layer = base_model.layers[-2].output
img_last_layer = BatchNormalization()(last_layer)

fusion = concatenate([text_last_layer, img_last_layer])
x = BatchNormalization()(fusion)
x = Dense(512, activation='relu')(x)
x = Dropout(.3)(x)
x = BatchNormalization()(x)
out = Dense(1, activation='softmax')(x)
multi_model = Model([model.inputs, image_input], out)


####


# https://github.com/artelab/Image-and-Text-fusion-for-UPMC-Food-101-using-BERT-and-CNNs/blob/main/stacking_early_fusion_UPMC_food101.ipynb
# !Not currently used
def add_dense_layers(model, parameters):
    model.add(AveragePooling2D(pool_size=(8, 8), name='AVG_Pooling'))
    model.add(Dropout(.4, name='Dropout_0.4'))
    model.add(Flatten(name='Flatten'))
    model.add(Dense(128, name='Dense_128'))

    # Keep model layers trainable
    for layer in model.layers:
        layer.trainable = True
    return model
