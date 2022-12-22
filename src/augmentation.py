#Training HSI Augmentations 
from torchvision import transforms

def train_augmentation():
    """Torchvision transforms
    Args:
        image: a torch vision tensor with C, H, W order
    Returns:
        transformed_image: an augmentated image
    """
    
    transform_list = []
    transform_list.append(transforms.RandomHorizontalFlip(p=1))
    transform_list.append(transforms.RandomVerticalFlip(p=1))
    
    return transforms.Compose(transform_list)
    
