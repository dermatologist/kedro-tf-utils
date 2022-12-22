"""Project pipelines."""

import tensorflow_text as text
# TODO: https://github.com/tensorflow/hub/issues/705
# ! Remove later


from typing import Dict

from kedro.pipeline import Pipeline, pipeline

from kedro_tf_utils.pipelines.fusion.pipeline import create_tabular_pipeline, create_text_fusion_pipeline
from kedro_tf_utils.pipelines.train_mm_simple.pipeline import create_train_pipeline

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    return {
        "__default__": create_tabular_pipeline(),
        "report": create_text_fusion_pipeline(),
        "train": create_train_pipeline()
        }

