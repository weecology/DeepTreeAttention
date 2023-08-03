#Training HSI Augmentations 
from torchvision import transforms
from src.utils import ZeroPad

def augment(train, image_size, pad_or_resize="pad"):
    """Torchvision transforms
    Args:
        image: a torch vision tensor with C, H, W order
        pad_or_resize: a str of "pad" or "resize"
    Returns:
        transformed_image: an augmentated image
    """
    transform_list = []

    if pad_or_resize == "pad":
        transform_list.append(ZeroPad(target_size=image_size))
    elif pad_or_resize =="resize":
        transform_list.append(transforms.Resize(size=(image_size,image_size), interpolation=transforms.InterpolationMode.NEAREST))
    if train:
        transform_list.append(transforms.RandomHorizontalFlip(p=1))
        transform_list.append(transforms.RandomVerticalFlip(p=1))
    
    return transforms.Compose(transform_list)
    
