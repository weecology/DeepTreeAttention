# Test fixmatch dataloader
from src import fixmatch
import torch


def test_TreeDataset(dm, config):
    config["preload_images"] = True
    ds = fixmatch.FixmatchDataset(df=dm.train.reset_index(drop=True), config=config)
    batch = ds[0]
    individaul, inputs = batch
    
    assert list(inputs.keys()) == ["HSI","Strong","Weak"]
    for index in range(len(inputs)):
        assert not torch.equal(inputs["Weak"][index], inputs["Strong"][index])
    