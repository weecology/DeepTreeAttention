#Test Model
from src.models import Hang2020
import torch

def test_Hang2020():
    m = Hang2020.Hang2020(bands=3, classes=10)
    image = torch.randn(20, 3, 100, 100)
    output = m(image)
    assert output.shape[0] == 10