"""
-- https://github.com/erdememekligil/oop-tensorflow-serving/blob/main/model/image_classifier_model.py
@author: erdememekligil
"""
import tensorflow as tf
from keras.models import Model
import logging
logger = logging.getLogger(__name__)



class ServingWrapperModel(tf.keras.Model):
    """
    SavedModel of this class will have two serving signatures. The default one (serving_default) calculates predictions
    using images as 4d arrays. The other signature, serving_bytes, operates on base64 encoded image bytes.

    See usage in :func:`kedro_tf_utils.pipelines.train.nodes.train_multimodal`
    """

    def __init__(self, base_model: Model = None, *args, **kwargs):
        inp, out, input_signature_bytes, input_signature_array = self.create_model_io(base_model=base_model)
        kwargs = {**kwargs, **{"inputs": inp, "outputs": out}}
        super(ServingWrapperModel, self).__init__(*args, **kwargs)
        self.model = base_model
        self.inp = inp
        self.out = out
        self.input_signature_bytes = input_signature_bytes
        self.input_signature_array = input_signature_array
        self.image_input_shape = [None, 224, 224, 3]
        self.predict_bytes_image = tf.function(input_signature=input_signature_bytes)(self._predict_bytes_image)
        self.predict_numpy_image = tf.function(input_signature=input_signature_array)(self._predict_numpy_image)

    def create_model_io(self, base_model: Model = None):
        inp = base_model.input
        out = base_model.output
        input_signature_bytes = []
        input_signature_array = []
        for input in base_model.inputs:
            if input.name == "input_1":
                self.image_input_shape = input.shape
                input_signature_bytes.append(tf.TensorSpec(
                    name="input_bytes", shape=(None,), dtype=tf.string))
            else:
                input_signature_bytes.append(tf.TensorSpec(
                    name=input.name, shape=input.shape, dtype=input.dtype))
            input_signature_array.append(tf.TensorSpec(
                name=input.name, shape=input.shape, dtype=input.dtype))
        logger.info(f"Input signature bytes: {input_signature_bytes}")
        logger.info(f"Input signature array: {input_signature_array}")
        return inp, out, input_signature_bytes, input_signature_array

    def call(self, inputs, training=None, mask=None):
        return super(ServingWrapperModel, self).call(inputs, training=training, mask=mask)

    def get_config(self):
        return super(ServingWrapperModel, self).get_config()

    def save(self, filepath, overwrite=True, include_optimizer=True, save_format=None, signatures=None, options=None,
             save_traces=True):
        """
        Saves model with custom signatures.
        serving_default: predict using 4 array image (numpy/tensor).
        serving_bytes: predict using base64 encoded image bytes.
        """
        if signatures is None:
            signatures = dict()
        signatures["serving_default"] = self.predict_numpy_image
        signatures["serving_bytes"] = self.predict_bytes_image
        super(ServingWrapperModel, self).save(filepath, overwrite, include_optimizer,
                     save_format, signatures, options, save_traces)

    def get_signatures(self):
        """
        Returns signatures of the model.
        """
        signatures = {
            "serving_default": self.predict_numpy_image,
            "serving_bytes": self.predict_bytes_image
        }
        return signatures

    def _predict_bytes_image(self, *args):
        """
        Predict using encoded image bytes.
        Other inputs are passed as they are.
        :param image: png, jpeg, bmp, gif encoded image bytes.
        :return: prediction result.
        """
        _args = list(args)
        for idx, tensor in enumerate(args):
            logger.info(f"Tensor name: {tensor.name}")
            if "_bytes" in tensor.name:
                logger.info(f"Tensor shape for image: {tensor.shape}")
                [height, width, color_channels] = self.image_input_shape[1:]
                input_tensor = tf.reshape(tensor, [])
                input_tensor = tf.image.decode_image(input_tensor, channels=color_channels)

                # Convert image to float and bring values in the range of 0-1
                input_tensor = tf.image.convert_image_dtype(input_tensor, dtype=tf.float32)

                # Reshape and add "batch" dimension (this expects a single image NOT in a list)
                input_tensor = tf.reshape(input_tensor, [height, width, color_channels])
                input_tensor = tf.expand_dims(input_tensor, 0, name="input_1")
                logger.info(f"Tensor shape for image after decoding is: {input_tensor.shape} and name is {input_tensor.name}")
                _args[idx] = input_tensor
            else:
                # ! Batch size should be 1 for all inputs for prediction
                # * This sets batch size to 1 for all inputs, ie (None,1) -> (1,)
                input_tensor = tf.reshape(tensor, [])
                name = tensor.name.split(":")[0]
                input_tensor = tf.expand_dims(input_tensor, 0, name=name)
            _args[idx] = input_tensor
        logger.info(f"Args  : {_args}")
        return self.call(tuple(_args))

    def _predict_numpy_image(self, *args):
        """
        Predict using 4d array image (numpy/tensor).
        Other inputs are passed as they are.
        :param inputs: 4d array image (batch, height, width, channel).
        :return: prediction result.
        """
        _args = list(args)
        for idx, tensor in enumerate(args):
            logger.info(f"Tensor name: {tensor.name}")
            if "input_1" in tensor.name:
                logger.info(f"Tensor shape for image: {tensor.shape}")
                [height, width, color_channels] = self.image_input_shape[1:]
                # Reshape and add "batch" dimension (this expects a single image NOT in a list)
                input_tensor = tf.reshape(input_tensor, [height, width, color_channels])
                input_tensor = tf.expand_dims(input_tensor, 0, name="input_1")
                logger.info(
                    f"Tensor shape for image after decoding is: {input_tensor.shape} and name is {input_tensor.name}")
                _args[idx] = input_tensor
            else:
                # ! Batch size should be 1 for all inputs for prediction
                # * This sets batch size to 1 for all inputs, ie (None,1) -> (1,)
                input_tensor = tf.reshape(tensor, [])
                name = tensor.name.split(":")[0]
                input_tensor = tf.expand_dims(input_tensor, 0, name=name)
            _args[idx] = input_tensor
        logger.info(f"Args  : {_args}")
        return self.call(tuple(_args))
