#Training HSI Augmentations 
from torchvision import transforms
from src.utils import resize_or_pad_image

def train_augmentation(train, image_size):
    """Torchvision transforms
    Args:
        image: a torch vision tensor with C, H, W order
    Returns:
        transformed_image: an augmentated image
    """
    
    transform_list = []

    transform_list.append(resize_or_pad_image(image_size=image_size))
    if train:
        transform_list.append(transforms.RandomHorizontalFlip(p=1))
        transform_list.append(transforms.RandomVerticalFlip(p=1))
    
    return transforms.Compose(transform_list)
    
