The model can be expanded by using multiple parallel convolutional neural networks that read the source document using different kernel sizes. This, in effect, creates a multichannel convolutional neural network for text that reads text with different n-gram sizes (groups of words).

### Copy vocab.txt to assets for fusion to work


## Adding serving layer references

* https://github.com/tensorflow/serving/issues/1869
* https://stackoverflow.com/questions/49492255/how-to-replace-or-insert-intermediate-layer-in-keras-model
* https://gist.github.com/dermatologist/1ff8ce47fbd0e69cf9104abc15665a9d