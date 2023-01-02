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


class TestFusionPipeline:
    def test_project_path(self, project_context):
        assert project_context.project_path == Path.cwd()

    def test_fusion_model(self, project_context):
        tabular_path = "data/06_models/tabular_model"
        image_path = 'data/06_models/imageset'
        _tabular_model = TensorFlowModelDataset(filepath=tabular_path)
        _image_model = TensorFlowModelDataset(filepath=image_path)
        tabular_model = _tabular_model.load()
        image_model = _image_model.load()
        conf_params = project_context.config_loader.get('**/fusion.yml')
        fusion_input = {
            'parameters': conf_params['fusion'],
            'tabular_model': tabular_model,
            'image_model': image_model
        }
        data = fusion(**fusion_input)
        assert data is not None


