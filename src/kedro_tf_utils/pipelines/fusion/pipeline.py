"""
This is a boilerplate pipeline 'fusion'
generated using Kedro 0.18.1
"""

from kedro.pipeline import Pipeline, node, pipeline
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline
from kedro_tf_text.pipelines.bert.pipeline import download_bert
from kedro_tf_text.pipelines.cnn.pipeline import cnn_text_pipeline
from kedro_tf_text.pipelines.preprocess.pipeline import (
    glove_embedding,
    process_text_pipeline,
)
from kedro_tf_text.pipelines.tabular.pipeline import tabular_model_pipeline

from kedro_tf_utils.pipelines.fusion.nodes import fusion


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([])

def create_fusion_pipeline(**kwargs) -> Pipeline:
    inputs = {}
    for name, data in kwargs.items():
        inputs[name] = data
    if not inputs:
        inputs = {"params:fusion": "params:fusion", "bert_model_saved": "bert_model_saved", "tabular_model_saved": "tabular_model_saved"}
    return pipeline([
                    node(
                        fusion,
                        # params first followed by the models
                        # ! Parameters come first followed by the models. Note this when using this node in the pipeline
                        inputs=inputs,
                        outputs="fusion_model",
                        name="create_fusion_model"
                    ),
                    ])

## Text + Image
## Example fusion models. First is parameters, then the models prefixed with the model type
fusion_inputs = {
    "parameters": "params:fusion",
    "text_model": "cnn_text_model",
    "image_model": "chexnet_model",
}
early_fusion_mm_pipeline = create_fusion_pipeline(**fusion_inputs)

# Demonstrates the use of modular pipelines: https://kedro.readthedocs.io/en/stable/nodes_and_pipelines/modular_pipelines.html
def create_text_fusion_pipeline(**kwargs) -> Pipeline:
    text_fusion_pipeline = modular_pipeline(pipe=glove_embedding, parameters={
                                            "params:embedding": "params:fusion"})
    _preprocess_text_pipeline = modular_pipeline(pipe=process_text_pipeline, parameters={
        "params:embedding": "params:fusion"})
    _cnn_text_pipeline = modular_pipeline(pipe=cnn_text_pipeline, parameters={
        "params:cnn_text_model": "params:fusion"})
    return _preprocess_text_pipeline + text_fusion_pipeline + _cnn_text_pipeline + early_fusion_mm_pipeline


# Tabular + Image
# Example fusion models. First is parameters, then the models prefixed with the model type
tabular_fusion_inputs = {
    "parameters": "params:fusion",
    "tabular_model": "tabular_model",
    "image_model": "chexnet_model",
}
tabular_early_fusion_mm_pipeline = create_fusion_pipeline(**tabular_fusion_inputs)

# Demonstrates the use of modular pipelines: https://kedro.readthedocs.io/en/stable/nodes_and_pipelines/modular_pipelines.html


def create_tabular_fusion_pipeline(**kwargs) -> Pipeline:
    _tabular_model_pipeline = modular_pipeline(pipe=tabular_model_pipeline, parameters={
                                            "params:tabular": "params:fusion"})

    return _tabular_model_pipeline + tabular_early_fusion_mm_pipeline


# Bert + Image
# Example fusion models. First is parameters, then the models prefixed with the model type
bert_fusion_inputs = {
    "parameters": "params:fusion",
    "bert_model": "bert_model_saved",
    "image_model": "chexnet_model",
}
bert_early_fusion_mm_pipeline = create_fusion_pipeline(**bert_fusion_inputs)

# Demonstrates the use of modular pipelines: https://kedro.readthedocs.io/en/stable/nodes_and_pipelines/modular_pipelines.html


def create_bert_fusion_pipeline(**kwargs) -> Pipeline:
    _bert_model_pipeline = modular_pipeline(pipe=download_bert, parameters={
        "params:bert_model": "params:fusion"})

    return _bert_model_pipeline + bert_early_fusion_mm_pipeline
