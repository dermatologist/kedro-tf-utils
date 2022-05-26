"""
This is a boilerplate pipeline 'fusion'
generated using Kedro 0.18.1
"""

from kedro.pipeline import Pipeline, node, pipeline
from kedro_tf_utils.pipelines.fusion.nodes import early_fusion_mm
from kedro_tf_text.pipelines.preprocess.nodes import tabular_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([])


def create_fusion_pipeline(datasets={}, **kwargs) -> Pipeline:
    return pipeline([

                    node(
                        tabular_model,
                        [datasets.get("tabular_data", "tabular_data"),
                         datasets.get("parameters", "parameters")],
                        datasets.get("datasetinmemory", "datasetinmemory"),
                        name="create_text_model"
                    ),
                    node(
                        early_fusion_mm,
                        [datasets.get("datasetinmemory", "datasetinmemory"),
                         datasets.get("chexnet_model", "chexnet_model"), datasets.get("parameters", "parameters")],
                        datasets.get("fusion_model", "fusion_model"),
                        name="create_fusion_model"
                    ),

                    ])
