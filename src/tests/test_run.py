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
from deeptables.models.layers import dt_custom_objects
from kedro.extras.datasets.tensorflow import TensorFlowModelDataset
from kedro_tf_utils.pipelines.cnn_text_model.nodes import last_layer_normalized
from kedro.io import PartitionedDataSet

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


# The tests below are here for the demonstration purpose
# and should be replaced with the ones testing the project
# functionality
class TestProjectContext:
    def test_project_path(self, project_context):
        assert project_context.project_path == Path.cwd()

    # def test_tabular_model(self, project_context):
    #     modelpath = "data/06_models/tabular_model"
    #     load_args = dict()
    #     load_args["custom_objects"] = dt_custom_objects
    #     tf_model = TensorFlowModelDataset(filepath=modelpath, load_args=load_args)
    #     reloaded = tf_model.load()
    #     conf_params = project_context.config_loader.get('**/cnn_text_model.yml')
    #     tabular_last_layer = last_layer_normalized(reloaded)
    #     assert tabular_last_layer is not None

    def test_image_load(self, project_context):
        dataset = "kedro.extras.datasets.pillow.ImageDataSet"
        path = 'data/01_raw/imageset'
        filename_suffix = ".jpg"
        partitioned_dataset = PartitionedDataSet(dataset=dataset, path=path, filename_suffix=filename_suffix)
        reloaded = partitioned_dataset.load()
        print(reloaded.keys())
        assert reloaded is not None