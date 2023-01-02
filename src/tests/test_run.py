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

class TestProjectContext:
    def test_project_path(self, project_context):
        assert project_context.project_path == Path.cwd()

    def test_tabular_model(self, project_context):
        csvpath = "data/01_raw/test_dataset.csv"
        tfpath = "data/06_models/tabular_model"
        data_set = CSVDataSet(filepath=csvpath)
        save_args = {
            'save_format': 'tf'
        }
        tf_model = TensorFlowModelDataset(filepath=tfpath, save_args=save_args)
        reloaded = data_set.load()
        conf_params = project_context.config_loader.get('**/fusion.yml')
        data = tabular_model(reloaded, conf_params['fusion'])
        tf_model.save(data)
        assert data is not None

    def test_image_load(self, project_context):
        filepath = None
        architecture = "DenseNet121"
        load_args = {
            "class_num": 14
        }
        data_set = TfModelWeights(filepath=filepath, architecture=architecture, load_args=load_args)
        model = data_set.load()
        model_path = 'data/06_models/imageset'
        tf_model = TensorFlowModelDataset(filepath=model_path)
        tf_model.save(model)
        assert model is not None
