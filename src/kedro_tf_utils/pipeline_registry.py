"""Project pipelines."""

import tensorflow_text as text
# TODO: https://github.com/tensorflow/hub/issues/705
# ! Remove later


from typing import Dict

from kedro.pipeline import Pipeline, pipeline

from kedro_tf_utils.pipelines.fusion.pipeline import create_bert_fusion_pipeline, create_tabular_fusion_pipeline, create_text_fusion_pipeline
from kedro_tf_utils.pipelines.train.pipeline import create_train_pipeline
from kedro_tf_text.pipelines.preprocess.pipeline import process_text_pipeline, glove_embedding

from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline


_process_text_pipeline = modular_pipeline(pipe=process_text_pipeline, parameters={"params:embedding": "params:fusion"})
_glove_embedding = modular_pipeline(pipe=glove_embedding, parameters={"params:embedding": "params:fusion"})

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    return {
        "__default__": create_text_fusion_pipeline(),
        "tabular": create_tabular_fusion_pipeline(),
        "train": create_train_pipeline(),
        "bert": create_bert_fusion_pipeline(),
        }

