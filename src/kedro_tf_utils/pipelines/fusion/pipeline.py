"""
This is a boilerplate pipeline 'fusion'
generated using Kedro 0.18.1
"""

from kedro.pipeline import Pipeline, node, pipeline
from kedro_tf_utils.pipelines.cnn_text_model.nodes import create_cnn_model
from kedro_tf_utils.pipelines.fusion.nodes import early_fusion_mm
from kedro_tf_text.pipelines.preprocess.nodes import tabular_model, pickle_processed_text, json_processed_text, create_glove_embeddings

from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline
from kedro_tf_text.pipelines.preprocess.pipeline import create_glove_embedding_pipeline, pickle_processed_text_pipeline
from kedro_tf_text.pipelines.preprocess.nodes import get_tf_bert_model

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([])


def create_tabular_pipeline(**kwargs) -> Pipeline:
    return pipeline([
                    node(
                        get_tf_bert_model,
                        inputs=["bert_model", "params:bert_model"],
                        outputs="bert_model_saved",
                        name="build_bert_model"
                    ),
                    node(
                        tabular_model,
                        inputs=["tabular_data", "params:fusion"],
                        outputs="datasetinmemory",
                        name="create_text_model"
                    ),
                    node(
                        early_fusion_mm, #! Parameters come first followed by the models. Note this when using this node in the pipeline
                        inputs=["params:fusion", "datasetinmemory",
                                "bert_model_saved", "chexnet_model"],  # any number of inputs
                        outputs="fusion_model",
                        name="create_fusion_model"
                    ),

                    ])

def create_fusion_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            create_cnn_model,
            inputs=["glove_embedding", "params:fusion"],
            outputs="datasetinmemory",
            name="create_cnn_model"
        ),
        node(
            early_fusion_mm,
            inputs=["datasetinmemory", "chexnet_model", "params:fusion"],
            outputs="fusion_model",
            name="create_fusion_model"
        ),

    ])

# Demonstrates the use of modular pipelines: https://kedro.readthedocs.io/en/stable/nodes_and_pipelines/modular_pipelines.html
def create_text_fusion_pipeline(**kwargs) -> Pipeline:
    created_glove = create_glove_embedding_pipeline()
    created_pickle = pickle_processed_text_pipeline()
    created_fusion = create_fusion_pipeline()
    glove_embedding_pipeline = modular_pipeline(pipe=created_glove, parameters={
        "params:embedding": "params:fusion"})
    processed_text_pipeline = modular_pipeline(pipe=created_pickle, parameters={
                                               "params:embedding": "params:fusion"})
    return processed_text_pipeline + glove_embedding_pipeline + created_fusion
