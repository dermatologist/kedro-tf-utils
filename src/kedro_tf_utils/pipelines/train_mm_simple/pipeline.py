"""
This is a boilerplate pipeline 'train_mm_simple'
generated using Kedro 0.18.4
"""

"""
This is a boilerplate pipeline 'train'
generated using Kedro 0.18.1
"""




from kedro.pipeline import Pipeline, node, pipeline
from kedro_tf_utils.pipelines.train_mm_simple.nodes import train_multimodal
def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([])


def create_train_pipeline(**kwargs) -> Pipeline:
    return pipeline([

                    node(
                        func=train_multimodal,
                        inputs=["tabular_data", "image_data",
                                "fusion_model", "params:train_model"],
                        outputs="trained_model",
                        name="train_model"
                    ),
                    ])
