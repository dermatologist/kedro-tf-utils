signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['age'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 1)
        name: serving_default_age:0
    inputs['bp'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 1)
        name: serving_default_bp:0
    inputs['gender'] tensor_info:
        dtype: DT_STRING
        shape: (-1, 1)
        name: serving_default_gender:0
    inputs['input_1'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 224, 224, 3)
        name: serving_default_input_1:0
    inputs['text_input_for_bert'] tensor_info:
        dtype: DT_STRING
        shape: (-1)
        name: serving_default_text_input_for_bert:0
    inputs['zip'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 1)
        name: serving_default_zip:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['fusion_1'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 1)
        name: StatefulPartitionedCall_2:0
  Method name is: tensorflow/serving/predict


  [tf.TensorSpec(name="image_bytes_string", shape=None, dtype=tf.string)]



## References

There are at least two ways:

def test(x):
    print "arg was "+str(x)
toDoList = []
args = ["hello"]
toDoList.append(lambda:test(*args))
# doesn't run yet
# run it
for f in toDoList:
    f()
If you think you might want to inspect or change the args before running, this next one is better:

def test(x):
    print "arg was "+str(x)
toDoList = []
args = ["hello"]
toDoList.append({'f': test, 'a': args})
# to run
for item in toDoList:
     item['f'](*item['a'])



https://stackoverflow.com/questions/54642590/add-metadata-to-tensorflow-frozen-graph-pb


class MyModel(tf.keras.Model):

  def __init__(self, metadata, **kwargs):
    super(MyModel, self).__init__(**kwargs)
    self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
    self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
    self.metadata = tf.constant(metadata)

  def call(self, inputs):
    x = self.dense1(inputs)
    return self.dense2(x)

  @tf.function(input_signature=[])
  def get_metadata(self):
    return self.metadata

model = MyModel('metadata_test')
input_arr = tf.random.uniform((5, 5, 1)) # This call is needed so Keras knows its input shape. You could define manually too
outputs = model(input_arr)
Then you can save and load your model as follows:

tf.keras.models.save_model(model, 'test_model_keras')
model_loaded = tf.keras.models.load_model('test_model_keras')


* https://stackoverflow.com/questions/56659949/saving-a-tf2-keras-model-with-custom-signature-defs

he solution is to create a tf.Module with functions for each signature definition:

class MyModule(tf.Module):
  def __init__(self, model, other_variable):
    self.model = model
    self._other_variable = other_variable

  @tf.function(input_signature=[tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32)])
  def score(self, waveform):
    result = self.model(waveform)
    return { "scores": results }

  @tf.function(input_signature=[])
  def metadata(self):
    return { "other_variable": self._other_variable }
And then save the module (not the model):

module = MyModule(model, 1234)
tf.saved_model.save(module, export_path, signatures={ "score": module.score, "metadata": module.metadata })