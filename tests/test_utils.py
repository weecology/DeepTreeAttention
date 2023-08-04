from src import utils
import pytest
import torch
from torchvision.transforms import functional as F

# One test for larger, one for smaller.
@pytest.mark.parametrize("image_size",[32, 10])
def test_zero_pad(zero_pad):
    # Create a sample image tensor
    sample = torch.randn((3, 16, 16))

    # Apply zero padding using the ZeroPad object
    padded_image = utils.ZeroPad(target_size=32)

    # Get the padded image size
    _, img_height, img_width = padded_image.size()

    # Check if the size matches the target size (32x32)
    assert img_height == 32
    assert img_width == 32