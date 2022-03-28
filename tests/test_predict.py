#test_predict
from src import predict
from src.models import dead
from skimage import io

#def test_dead_tree_model(dead_model_path, config, ROOT):
    #m = dead.AliveDead.load_from_checkpoint(dead_model_path, config=config)
    #m.eval()
    #dead_tree = io.imread("{}/tests/data/dead_tree.png".format(ROOT))
    #transform = dead.get_transform(augment=False)
    #dead_tree_transformed = transform(dead_tree)
    #score = m(dead_tree_transformed.unsqueeze(0)).detach()
    ##assert np.argmax(score) == 1
    
    ##dead_tree = io.imread("{}/tests/data/dead_tree2.png".format(ROOT))
    ##transform = dead.get_transform(augment=False)
    ##dead_tree_transformed = transform(dead_tree)
    ##score = m(dead_tree_transformed.unsqueeze(0)).detach()
    ##assert np.argmax(score) == 1
    
    ##alive_tree = io.imread("{}/tests/data/alive_tree.png".format(ROOT))
    ##alive_tree = transform(alive_tree)
    ##score = m(alive_tree.unsqueeze(0)).detach()
    ##assert np.argmax(score) == 0
    
def test_predict_tile(species_model_path, config, ROOT):
    PATH =  "{}/tests/data/2019_D01_HARV_DP3_726000_4699000_image_crop.tif".format(ROOT)
    config["CHM_pool"] = None
    trees = predict.predict_tile(PATH, dead_model_path = None, species_model_path=species_model_path, config=config)
    assert all([x in trees.columns for x in ["pred_taxa_top1","geometry","top1_score"]])