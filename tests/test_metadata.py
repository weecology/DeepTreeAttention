#Test metadata model
from src.models import metadata
import torch

def test_metadata():
    m = metadata.metadata(sites=23, classes=10)
    sites = torch.zeros(20, 23)
    for x in range(sites.shape[0]):
        sites[x,torch.randint(low=0,high=23,size=(1,))] =1         
    output = m(sites)
    assert output.shape == (20,10)