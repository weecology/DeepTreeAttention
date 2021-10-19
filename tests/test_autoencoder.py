#test autoencoder
from src.models import autoencoder
import torch
import os
import pytest
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def test_conv_module():
    m = autoencoder.conv_module(in_channels=369, filters=32)
    image = torch.randn(20, 369, 11, 11)
    output = m(image)
    assert output.shape == (20,32,11,11)
    
def test_autoencoder():
    m = autoencoder.autoencoder(bands=369, classes=10)
    image = torch.randn(20, 369, 11, 11)
    output = m(image)    
    assert output.shape == (20,10)
    