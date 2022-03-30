#test_predict
import os
from src import predict
from src.models import dead
from src import main
from skimage import io

#def test_dead_tree_model(dead_model_path, config, ROOT):
    #m = dead.AliveDead.load_from_checkpoint(dead_model_path, config=config)
    #m.eval()
    #dead_tree = io.imread("{}/tests/data/dead_tree.png".format(ROOT))
    #transform = dead.get_transform(augment=False)
    #dead_tree_transformed = transform(dead_tree)
    #score = m(dead_tree_transformed.unsqueeze(0)).detach()
    
def test_predict_tile(species_model_path, config, ROOT):
    HSI_paths = {}
    HSI_paths["2018"] = "{}/tests/data/2018_D01_HARV_DP3_726000_4699000_image_crop_2018.tif".format(ROOT)
    HSI_paths["2019"] = "{}/tests/data/2019_D01_HARV_DP3_726000_4699000_image_crop_2019.tif".format(ROOT)
    
    config["CHM_pool"] = None
    species_model_dir = os.path.dirname(species_model_path)
    trees = predict.predict_tile(
        HSI_paths,
        dead_model_path=None,
        species_model_dir=species_model_dir,
        config=config)
    assert all([x in trees.columns for x in ["pred_taxa_top1","geometry","top1_score"]])