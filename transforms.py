from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2



# 这个函数的作用是构建用于训练和测试过程中的图像数据转换（data transformation），确保输入模型的数据具有一致性和可训练性。这个函数接受以下参数：
# height 和 width：目标图像的高度和宽度。
# max_pixel_value：图像像素值的最大值，通常为255。
# norm_mean 和 norm_std：图像标准化的均值和标准差。如果未提供这些参数，函数将使用默认的 ImageNet 数据集的均值和标准差。
# **kwargs：其他可能的参数。
def build_transforms(height, width, max_pixel_value=255.0, norm_mean=[0.485, 0.456, 0.406],
                     norm_std=[0.229, 0.224, 0.225], **kwargs):
    """Builds train and test transform functions.

    Args:
        height (int): target image height.
        width (int): target image width.E
        norm_mean (list or None, optional): normalization mean values. Default is ImageNet means.
        norm_std (list or None, optional): normalization standard deviation values. Default is
            ImageNet standard deviation values.
        max_pixel_value (float): max pixel value
    """

    if norm_mean is None or norm_std is None:
        norm_mean = [0.485, 0.456, 0.406] # imagenet mean
        norm_std = [0.229, 0.224, 0.225] # imagenet std
    normalize = transforms.Normalize(mean=norm_mean, std=norm_std)

    # train_transform，用于训练集的数据转换。这个转换通常包括随机水平翻转、高斯噪声、高斯模糊、调整图像大小和颜色归一化等操作。最后，将图像转换为张量形式。
    train_transform = A.Compose([
        A.HorizontalFlip(),
        A.GaussNoise(p=0.1),
        A.GaussianBlur(p=0.1),
        A.Resize(height, width),
        A.Normalize(mean=norm_mean, std=norm_std, max_pixel_value=max_pixel_value),
        ToTensorV2(),
    ])
    # test_transform：用于测试集的数据转换。这个转换通常只包括调整图像大小和颜色归一化，并将图像转换为张量形式。
    test_transform = A.Compose([
        A.Resize(height, width),
        A.Normalize(mean=norm_mean, std=norm_std, max_pixel_value=max_pixel_value),
        ToTensorV2(),
    ])
    # 这样，通过这个函数构建的数据转换对象可以在训练和测试过程中对图像数据进行相应的处理和标准化，以便输入到深度学习模型中进行训练和评估。

    return train_transform, test_transform






