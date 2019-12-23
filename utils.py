import numpy as np
from matplotlib import pyplot as plt
import colorcet as cc
from PIL import Image
import torchvision


def normalize(mask, percentile=99):
    vmax = np.percentile(mask, percentile)
    vmin = np.percentile(mask, 1)
    return np.clip((mask - vmin) / (vmax - vmin), 0, 1)


def show_mask(mask, cmap=None, alpha=None):
    if cmap == cc.cm.gray and len(mask) > 2:
        mask = np.sum(mask, axis=2)
        mask = normalize(mask)

    plt.imshow(mask, cmap=cmap, alpha=alpha, interpolation='lanczos')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def cut_image_with_mask(image_path, mask, percentile=80):
    image = np.moveaxis(load_image(image_path, size=mask.shape[0], preprocess=False).numpy(), 0, -1)
    mask = mask > np.percentile(mask, percentile)
    image[~mask] = 0

    plt.imshow(image, interpolation='lanczos')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def show_mask_on_image(image_path, mask, cmap=cc.cm.bmy, alpha=0.7):
    image = load_image(image_path, size=mask.shape[0], color_mode='L', preprocess=False).numpy().squeeze()
    plt.imshow(image, cmap=cc.cm.gray, interpolation='lanczos')
    show_mask(mask, cmap, alpha)


def pil_loader(path, color_mode='RGB'):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert(color_mode)


def load_image(path, size=None, color_mode='RGB', preprocess=True):
    pil_image = pil_loader(path, color_mode)
    shape = np.array(pil_image).shape
    transform_list = list()
    if size is not None and size != shape[0]:
        if size < shape[0]:
            if size < 256 < shape[0]:
                transform_list.append(torchvision.transforms.Resize(256))
            transform_list.append(torchvision.transforms.CenterCrop(size))
        else:
            print(f"Warning: Desired size {size} larger than image size {shape[0]}x{shape[1]}. Upscaling.")
            transform_list.append(torchvision.transforms.Resize(size))
    transform_list.append(torchvision.transforms.ToTensor())
    if preprocess:
        transform_list.append(torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    transform = torchvision.transforms.Compose(transform_list)
    return transform(pil_image)
