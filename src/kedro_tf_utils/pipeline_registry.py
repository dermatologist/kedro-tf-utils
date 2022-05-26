"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline, pipeline

from kedro_tf_utils.pipelines.fusion.pipeline import create_fusion_pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    return {"__default__": create_fusion_pipeline()}
