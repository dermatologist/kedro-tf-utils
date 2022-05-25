"""
This is a boilerplate pipeline 'cnn_text_model'
generated using Kedro 0.18.1
"""

from kedro.pipeline import Pipeline, node, pipeline
from kedro_tf_text.pipelines.preprocess.nodes import tabular_model

from kedro_tf_utils.pipelines.cnn_text_model.nodes import early_fusion_mm

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([])


def create_fusion_pipeline(**kwargs) -> Pipeline:
    return pipeline([

                    node(
                        tabular_model,
                        ["tabular_data", "parameters"],
                        "datasetinmemory",
                        name="create_text_model"
                    ),
                    node(
                        early_fusion_mm,
                        ["datasetinmemory", "chexnet_model", "parameters"],
                        "fusion_model",
                        name="create_fusion_model"
                    ),

    ])