#Test Model
from src.models import Hang2020
import torch
import os
import pytest
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def test_conv_module():
    m = Hang2020.conv_module(in_channels=369, filters=32)
    image = torch.randn(20, 369, 11, 11)
    output = m(image)
    assert output.shape == (20,32,11,11)

def test_conv_module_maxpooling():
    m = Hang2020.conv_module(in_channels=32, filters=64, maxpool_kernel=(2,2))
    image = torch.randn(20, 32, 11, 11)
    output = m(image, pool = True)
    assert output.shape == (20,64,5,5)

@pytest.mark.parametrize("conv_dimension",[(20,32,11,11),(20,64,5,5),(20,128,2,2)])
def test_spatial_attention(conv_dimension):
    """Check spectral attention for each convoutional dimension"""
    m = Hang2020.spatial_attention(filters=conv_dimension[1], classes=10)
    image = torch.randn(conv_dimension)
    attention, scores = m(image)
    assert scores.shape[0] == 10
    
@pytest.mark.parametrize("conv_dimension",[(20,32,11,11),(20,64,5,5),(20,128,2,2)])
def test_spectral_attention(conv_dimension):
    """Check spectral attention for each convoutional dimension"""
    m = Hang2020.spectral_attention(filters=conv_dimension[1], classes=10)
    image = torch.randn(conv_dimension)
    attention, scores = m(image)
    assert scores.shape[0] == 10
    
def test_spectral_network():
    m = Hang2020.spectral_network(bands=369, classes=10)
    image = torch.randn(20, 369, 11, 11)
    output = m(image)
    assert len(output) == 3
    assert output[0].shape[0] == 10
    
def test_spatial_network():
    m = Hang2020.spatial_network(bands=369, classes=10)
    image = torch.randn(20, 369, 11, 11)
    output = m(image)
    assert len(output) == 3
    assert output[0].shape[0] == 10