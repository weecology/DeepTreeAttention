#test autoencoder
from src.models import autoencoder
from pytorch_lightning import Trainer
from src import generate
import geopandas as gpd

def test_autoencoder(ROOT, config, rgb_path, tmpdir):
    data_path = "{}/tests/data/crown.shp".format(ROOT)
    gdf = gpd.read_file(data_path)
    gdf["RGB_tile"] = rgb_path
    annotations = generate.generate_crops(
        gdf=gdf, rgb_glob="{}/tests/data/*.tif".format(ROOT),
        convert_h5=False, sensor_glob="{}/tests/data/*.tif".format(ROOT), savedir=tmpdir)
    annotations = annotations.reset_index(drop=True)
    model = autoencoder.autoencoder(train_df=annotations, val_df=annotations, classes=3, config=config, comet_logger=None)
    trainer = Trainer(fast_dev_run=True)
    results = trainer.validate(model)
    