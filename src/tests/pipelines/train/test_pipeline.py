"""
This module contains an example test.

Tests should be placed in ``src/tests``, in modules that mirror your
project's structure, and in files named test_*.py. They are simply functions
named ``test_*`` which test a unit of logic.

To run the tests, run ``kedro test`` from the project root directory.
"""

from pathlib import Path

import pytest

from kedro.framework.project import settings
from kedro.config import ConfigLoader
from kedro.framework.context import KedroContext
from kedro.framework.hooks import _create_hook_manager
from kedro.extras.datasets.pandas import CSVDataSet
from kedro.extras.datasets.tensorflow import TensorFlowModelDataset

from kedro_tf_text.pipelines.tabular.nodes import tabular_model
from kedro_tf_image.extras.datasets.tf_model_weights import TfModelWeights

from kedro_tf_utils.pipelines.fusion.nodes import fusion
from kedro.io import PartitionedDataSet

from kedro_tf_utils.pipelines.train.nodes import train_multimodal


@pytest.fixture
def config_loader():
    return ConfigLoader(conf_source=str(Path.cwd() / settings.CONF_SOURCE))


@pytest.fixture
def project_context(config_loader):
    return KedroContext(
        package_name="kedro_tf_utils",
        project_path=Path.cwd(),
        config_loader=config_loader,
        hook_manager=_create_hook_manager(),
    )


class TestTrainPipeline:
    def test_project_path(self, project_context):
        assert project_context.project_path == Path.cwd()

    def test_fusion_model(self, project_context):

        fusion_path = 'data/06_models/fusion_model'
        _fusion_model = TensorFlowModelDataset(filepath=fusion_path)
        fusion_model = _fusion_model.load()
        csvpath = "data/01_raw/test_dataset.csv"
        tabular_data = CSVDataSet(filepath=csvpath).load()
        dataset = {
            "type": "kedro_tf_image.extras.datasets.tf_image_dataset.TfImageDataSet",
            "preprocess_input": "tensorflow.keras.applications.resnet50.preprocess_input",
            "imagedim": 224
        }
        path = 'data/01_raw/imageset'
        filename_suffix = ".jpg"
        data_set = PartitionedDataSet(
            dataset=dataset, path=path, filename_suffix=filename_suffix)
        image_data = data_set.load()

        conf_params = project_context.config_loader.get('**/fusion.yml')
        train_input = {
            'parameters': conf_params['fusion'],
            'model': fusion_model,
            'tabular_data': tabular_data,
            'image_data': image_data
        }


        with pytest.raises(ValueError):
            data = train_multimodal(**train_input)
