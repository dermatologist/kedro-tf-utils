"""
This is a boilerplate pipeline 'train'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline

from kedro_tf_utils.pipelines.train.nodes import train_multimodal


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([])

# ! Modify to take in the correct inputs


# def create_train_pipeline(**kwargs) -> Pipeline:
#     inputs = {}
#     for name, data in kwargs.items():
#         inputs[name] = data
#     if not inputs:
#         inputs = {"parameters": "params:train", "model": "fusion_model",
#                   "processed_text": "processed_text", "text_data": "text_data", "image_data": "image_data"}
#     return pipeline([
#                     node(
#                         train_multimodal,
#                         # params first followed by the models
#                         # ! Parameters come first followed by the models. Note this when using this node in the pipeline
#                         inputs=inputs,
#                         outputs="trained_model",
#                         name="create_trained_model"
#                     ),
#                     ])


# def create_train_pipeline(**kwargs) -> Pipeline:
#     inputs = {}
#     for name, data in kwargs.items():
#         inputs[name] = data
#     if not inputs:
#         inputs = {"parameters": "params:train", "model": "fusion_model",
#                   "tabular_data": "tabular_data", "image_data": "image_data"}
#     return pipeline([
#                     node(
#                         train_multimodal,
#                         # params first followed by the models
#                         # ! Parameters come first followed by the models. Note this when using this node in the pipeline
#                         inputs=inputs,
#                         outputs="trained_model",
#                         name="create_trained_model"
#                     ),
#                     ])


def create_train_pipeline(**kwargs) -> Pipeline:
    inputs = {}
    for name, data in kwargs.items():
        inputs[name] = data
    if not inputs:
        inputs = {"parameters": "params:train", "model": "fusion_model",
                  "bert_data": "text_data", "image_data": "image_data"}
    return pipeline([
                    node(
                        train_multimodal,
                        # params first followed by the models
                        # ! Parameters come first followed by the models. Note this when using this node in the pipeline
                        inputs=inputs,
                        outputs="trained_model",
                        name="create_trained_model"
                    ),
                    ])
