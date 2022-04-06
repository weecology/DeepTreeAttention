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
    
def RGB_augmentation():
    """Torchvision transforms
    Args:
        image: a torch vision tensor with C, H, W order
    Returns:
        transformed_image: an augmentated image
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    transform_list = []
    transform_list.append(normalize)
    transform_list.append(transforms.RandomHorizontalFlip(p=1))
    transform_list.append(transforms.RandomVerticalFlip(p=1))
    
    return transforms.Compose(transform_list)