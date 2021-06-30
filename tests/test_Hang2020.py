#Test Model
from src.models import Hang2020
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


#This test assume a crop size of 11 x 11 x 369.

def test_spatial_attention():
    m = Hang2020.spatial_attention(filters=32, classes=10)
    image = torch.randn(20, 32, 9, 9)
    output = m(image)
    assert output.shape[0] == 10
    
def test_Hang2020():
    m = Hang2020.Hang2020(bands=369, classes=10)
    image = torch.randn(20, 369, 11, 11)
    output = m(image)
    assert output.shape[0] == 10