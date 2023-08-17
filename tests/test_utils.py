from src import utils
import pytest
import torch
from torchvision.transforms import functional as F
from torch.utils.data.dataloader import default_collate

# One test for larger, one for smaller.
@pytest.mark.parametrize("image_size",[32, 10])
@pytest.mark.parametrize("sample_size",[(3, 16,16), (3, 16,8)])
def test_zero_pad(image_size, sample_size):
    # Create a sample image tensor
    sample = torch.randn(sample_size)

    # Apply zero padding using the ZeroPad object
    transformer = utils.ZeroPad(target_size=image_size)
    padded_image = transformer(sample)

    # Get the padded image size
    _, img_height, img_width = padded_image.size()

    # Check if the size matches the target size
    assert img_height == image_size
    assert img_width == image_size

def test_skip_none_collate():
    # Define test inputs
    batch = [
        (torch.tensor([1]), {"HSI": {2019: torch.tensor([1, 2, 3]), 2020: torch.tensor([4, 5, 6])}}),
        (torch.tensor([2]), {"HSI": {2019: torch.tensor([0, 0, 0]), 2020: torch.tensor([0, 0, 0])}}),
        (torch.tensor([3]), {"HSI": {2019: torch.tensor([10, 11, 12]), 2020: torch.tensor([0, 0, 0])}})
    ]

    # Call the function being tested
    result = utils.skip_none_collate(batch)

    # Define the expected output
    expected_output = [
        (torch.tensor([1]), {"HSI": {2019: torch.tensor([1, 2, 3]), 2020: torch.tensor([4, 5, 6])}}),
        (torch.tensor([3]), {"HSI": {2019: torch.tensor([10, 11, 12]), 2020: torch.tensor([0, 0, 0])}})
    ]

    # Check the output, should be length two, with samples 1 and 3
    assert len(result) == 2
    assert expected_output[0][0] == 1
    assert expected_output[1][0] == 3