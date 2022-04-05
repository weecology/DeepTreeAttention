#test RGB
from src.models import RGB
import torch

def test_RGB():
    m = RGB.RGB()
    image = torch.randn(20, 3, 110, 110)
    output = m(image)

def test_spectral_network():
    m = RGB.spectral_fusion_network(classes=10, bands=349)
    hsi_image = torch.randn(20, 349, 11, 11)
    rgb_image = torch.randn(20, 3, 110, 110)
    output = m(hsi_image, rgb_image)
    assert output.shape == (20,10)    