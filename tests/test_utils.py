from src import utils
import pytest
import torch
from torchvision.transforms import functional as F

# One test for larger, one for smaller.
@pytest.mark.parametrize("image_size",[32, 10])
def test_zero_pad(image_size):
    # Create a sample image tensor
    sample = torch.randn((3, 16, 16))

    # Apply zero padding using the ZeroPad object
    transformer = utils.ZeroPad(target_size=image_size)
    padded_image = transformer(sample)

    # Get the padded image size
    _, img_height, img_width = padded_image.size()

    # Check if the size matches the target size
    assert img_height == image_size
    assert img_width == image_size