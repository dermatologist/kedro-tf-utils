"""
This is a boilerplate pipeline 'fusion'
generated using Kedro 0.18.1
"""

from kedro.pipeline import Pipeline, node, pipeline
from kedro_tf_utils.pipelines.cnn_text_model.nodes import create_cnn_model
from kedro_tf_utils.pipelines.fusion.nodes import early_fusion_mm
from kedro_tf_text.pipelines.preprocess.nodes import tabular_model, pickle_processed_text, json_processed_text, create_glove_embeddings


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


def create_text_fusion_pipeline(datasets={}, **kwargs) -> Pipeline:
    return pipeline([

                    node(
                        pickle_processed_text,
                        [datasets.get("text_data", "text_data"),
                         datasets.get("parameters", "parameters")],
                        datasets.get("processed_text", "processed_text"),
                        name="pickle_processed_text"
                    ),
                    node(
                        json_processed_text,
                        [datasets.get("text_data", "text_data"),
                         datasets.get("parameters", "parameters")],
                        datasets.get("vocab_json", "vocab_json"),
                        name="create_vocab"
                    ),
                    node(
                        create_glove_embeddings,
                        [datasets.get("pretrained_embedding", "pretrained_embedding"),
                         datasets.get("vocab_json", "vocab_json"), datasets.get("parameters", "parameters")],
                        datasets.get("glove_embedding", "glove_embedding"),
                        name="create_glove_embeddings"
                    ),
                    node(
                        create_cnn_model,
                        [datasets.get("glove_embedding", "glove_embedding"),
                         datasets.get("parameters", "parameters")],
                        datasets.get("datasetinmemory", "datasetinmemory"),
                        name="create_cnn_model"
                    ),
                    node(
                        early_fusion_mm,
                        [datasets.get("datasetinmemory", "datasetinmemory"),
                         datasets.get("chexnet_model", "chexnet_model"), datasets.get("parameters", "parameters")],
                        datasets.get("fusion_model", "fusion_model"),
                        name="create_fusion_model"
                    ),

                    ])
