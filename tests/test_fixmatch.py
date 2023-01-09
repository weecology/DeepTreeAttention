# Test fixmatch dataloader
from src import fixmatch
import cProfile, pstats


def test_TreeDataset(dm, config):
    config["preload_images"] = True
    ds = fixmatch.TreeDataset(df=dm.train.reset_index(drop=True), config=config)
    len(ds) == dm.train.shape[0]
    
    profiler = cProfile.Profile()
    profiler.enable()
    [x for x in ds]
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()