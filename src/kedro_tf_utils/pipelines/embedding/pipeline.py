"""
This is a boilerplate pipeline 'embedding'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline

from kedro_tf_utils.pipelines.embedding.nodes import create_embedding


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([])


def create_embedding_pipeline(**kwargs) -> Pipeline:
    inputs = {}
    outputs = "embedding"
    for name, data in kwargs.items():
        if name == "outputs":
            outputs = data
        else:
            inputs[name] = data
    if not inputs:
        inputs = {"parameters": "params:embedding", "model": "tabular_model",
                  "tabular_data": "tabular_data"}
    return pipeline([
                    node(
                        create_embedding,
                        # params first followed by model and data
                        # ! Parameters come first followed by the models. Note this when using this node in the pipeline
                        inputs=inputs,
                        outputs=outputs,
                        name="create_embedding"
                    ),
                    ])
