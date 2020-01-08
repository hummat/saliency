import numpy as np
from matplotlib import pyplot as plt
import colorcet as cc
from PIL import Image
import torchvision


def normalize(mask, vmin=None, vmax=None, percentile=99):
    if vmax is None:
        vmax = np.percentile(mask, percentile)
    if vmin is None:
        vmin = np.min(mask)
    return (mask - vmin) / (vmax - vmin + 1e-10)


def make_grayscale(mask):
    return np.sum(mask, axis=2)


def make_black_white(mask):
    return make_grayscale(np.abs(mask))


def show_mask(mask, title='', cmap=None, alpha=None, norm=True, axis=None):
    if norm:
        mask = normalize(mask)
    (vmin, vmax) = (-1, 1) if cmap == cc.cm.bkr else (0, 1)
    if axis is None:
        plt.imshow(mask, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax, interpolation='lanczos')
        if title:
            plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    else:
        axis.imshow(mask, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax, interpolation='lanczos')
        if title:
            axis.set_title(title)
        axis.axis('off')


def cut_image_with_mask(image_path, mask, title='', percentile=70, axis=None):
    image = np.moveaxis(load_image(image_path, size=mask.shape[0], preprocess=False).numpy().squeeze(), 0, -1)
    mask = mask > np.percentile(mask, percentile)
    image[~mask] = 0

    if axis is None:
        plt.imshow(image, interpolation='lanczos')
        if title:
            plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    else:
        axis.imshow(image, interpolation='lanczos')
        if title:
            axis.set_title(title)
        axis.axis('off')


def show_mask_on_image(image_path, mask, title='', cmap=cc.cm.bmy, alpha=0.7, axis=None):
    image = load_image(image_path, size=mask.shape[0], color_mode='L', preprocess=False).numpy().squeeze()
    if axis is None:
        plt.imshow(image, cmap=cc.cm.gray, interpolation='lanczos')
    else:
        axis.imshow(image, cmap=cc.cm.gray, interpolation='lanczos')
    show_mask(mask, title, cmap, alpha, norm=False, axis=axis)


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
    return transform(pil_image).unsqueeze(0)
