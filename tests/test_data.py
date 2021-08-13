#Test data module
from src import data
import os
from distributed import Client
ROOT = os.path.dirname(os.path.dirname(data.__file__))


def test_TreeData(tmpdir):
    #dask client to test
    client = Client()
    #One site's worth of data
    csv_file = "{}/tests/data/sample_neon.csv".format(ROOT)
    #Turn of CHM filtering for the moment
    config = data.read_config(config_path="{}/config.yml".format(ROOT))
    config["min_CHM_height"] = None
    config["iterations"] = 1
    data_module = data.TreeData(config=config, data_dir="{}/tests/data".format(ROOT))
    data_module.setup(csv_file=csv_file, regenerate=True, client=client)