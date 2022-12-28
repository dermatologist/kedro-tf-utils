"""
This is a boilerplate pipeline 'fusion'
generated using Kedro 0.18.1
"""

from kedro.pipeline import Pipeline, node, pipeline
from kedro_tf_utils.pipelines.fusion.nodes import early_fusion_mm

from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline
from kedro_tf_text.pipelines.preprocess.pipeline import word2vec_embedding
from kedro_tf_text.pipelines.bert.nodes import get_tf_bert_model
from kedro_tf_text.pipelines.tabular.nodes import tabular_model
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline
from kedro_tf_text.pipelines.cnn.pipeline import cnn_text_pipeline

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([])


def create_bert_pipeline(**kwargs) -> Pipeline:
    return pipeline([
                    node(
                        get_tf_bert_model,
                        inputs=["bert_model", "params:bert_model"],
                        outputs="bert_model_saved",
                        name="build_bert_model"
                    ),
                    ])

def create_tabular_pipeline(**kwargs) -> Pipeline:
    return pipeline([
                    node(
                        tabular_model,
                        inputs=["tabular_data", "params:fusion"],
                        outputs="tabular_model_saved",
                        name="create_tabular_model"
                    ),
                    ])

def create_fusion_pipeline(**kwargs) -> Pipeline:
    inputs = {}
    for name, data in kwargs.items():
        inputs[name] = data
    if not inputs:
        inputs = {"params:fusion": "params:fusion", "bert_model_saved": "bert_model_saved", "tabular_model_saved": "tabular_model_saved"}
    return pipeline([
                    node(
                        early_fusion_mm,
                        # params first followed by the models
                        # ! Parameters come first followed by the models. Note this when using this node in the pipeline
                        inputs=inputs,
                        outputs="fusion_model",
                        name="create_fusion_model"
                    ),
                    ])

# early_fusion_mm_pipeline = pipeline([
#     node(
#         early_fusion_mm,
#         # params first followed by the models
#         # ! Parameters come first followed by the models. Note this when using this node in the pipeline
#         inputs=["params:fusion", "cnn_text_model", "chexnet_model"],
#         outputs="fusion_model",
#         name="create_fusion_model"
#     ),
# ])

fusion_inputs = {
    "parameters": "params:fusion",
    "cnn_text_model": "cnn_text_model",
    "image_chexnet_model": "chexnet_model"
}
early_fusion_mm_pipeline = create_fusion_pipeline(**fusion_inputs)

# Demonstrates the use of modular pipelines: https://kedro.readthedocs.io/en/stable/nodes_and_pipelines/modular_pipelines.html
def create_text_fusion_pipeline(**kwargs) -> Pipeline:
    text_fusion_pipeline = modular_pipeline(pipe=word2vec_embedding, parameters={
                                            "params:embedding": "params:fusion"})
    _cnn_text_pipeline = modular_pipeline(pipe=cnn_text_pipeline, parameters={
        "params:cnn_text_model": "params:fusion"})
    return text_fusion_pipeline + _cnn_text_pipeline + early_fusion_mm_pipeline
