import numpy as np
from skimage import segmentation
from skimage.transform import resize
from skimage.morphology import disk, dilation

from saliency_mask import SaliencyMask
from integrated_gradients import IntegratedGradients


def _normalize_image(im, value_range, resize_shape=None):
    im_max = np.max(im)
    im_min = np.min(im)
    im = (im - im_min) / (im_max - im_min)
    im = im * (value_range[1] - value_range[0]) + value_range[0]
    if resize_shape is not None:
        im = resize(im, resize_shape, order=3, mode='constant', preserve_range=True, anti_aliasing=True)
    return im


def _unpack_segs_to_masks(segs):
    masks = []
    for seg in segs:
        for l in range(seg.min(), seg.max() + 1):
            masks.append(seg == l)
    return masks


def _get_segments_felzenszwalb(image, dilation_rad=5):
    original_shape = image.shape[:2]
    image = _normalize_image(image, value_range=(-1.0, 1.0), resize_shape=(224, 224))
    segs = []
    for scale in [50, 100, 150, 250, 500, 1200]:
        for sigma in [0.8]:
            seg = segmentation.felzenszwalb(image, scale=scale, sigma=sigma, min_size=150)
            seg = resize(seg, original_shape, order=0, preserve_range=True, mode='constant',
                         anti_aliasing=False).astype(np.int)
            segs.append(seg)
    masks = _unpack_segs_to_masks(segs)
    if dilation_rad:
        selem = disk(dilation_rad)
        masks = [dilation(mask, selem=selem) for mask in masks]
    return masks


def _get_diff_mask(add_mask, base_mask):
    return np.logical_and(add_mask, np.logical_not(base_mask))


def _get_diff_cnt(add_mask, base_mask):
    return np.sum(_get_diff_mask(add_mask, base_mask))


def _gain_density(mask1, attr, mask2=None):
    if mask2 is None:
        added_mask = mask1
    else:
        added_mask = _get_diff_mask(mask1, mask2)
    if not np.any(added_mask):
        return -np.inf
    else:
        return attr[added_mask].mean()


def _xrai(attr,
          segs,
          gain_fun=_gain_density,
          area_perc_th=1.0,
          min_pixel_diff=50,
          integer_segments=True):
    output_attr = -np.inf * np.ones(shape=attr.shape, dtype=np.float)

    n_masks = len(segs)
    current_area_perc = 0.0
    current_mask = np.zeros(attr.shape, dtype=bool)

    masks_trace = []
    remaining_masks = {ind: mask for ind, mask in enumerate(segs)}

    added_masks_cnt = 1
    # While the mask area is less than area_th and remaining_masks is not empty
    while current_area_perc <= area_perc_th:
        best_gain = -np.inf
        best_key = None
        remove_key_queue = []
        for mask_key in remaining_masks:
            mask = remaining_masks[mask_key]
            # If mask does not add more than min_pixel_diff to current mask, remove
            mask_pixel_diff = _get_diff_cnt(mask, current_mask)
            if mask_pixel_diff < min_pixel_diff:
                remove_key_queue.append(mask_key)
                continue
            gain = gain_fun(mask, attr, mask2=current_mask)
            if gain > best_gain:
                best_gain = gain
                best_key = mask_key
        for key in remove_key_queue:
            del remaining_masks[key]
        if len(remaining_masks) == 0:
            break
        added_mask = remaining_masks[best_key]
        mask_diff = _get_diff_mask(added_mask, current_mask)
        masks_trace.append((mask_diff, best_gain))

        current_mask = np.logical_or(current_mask, added_mask)
        current_area_perc = np.mean(current_mask)
        output_attr[mask_diff] = best_gain
        del remaining_masks[best_key]  # delete used key
        added_masks_cnt += 1

    uncomputed_mask = output_attr == -np.inf
    # Assign the uncomputed areas a value such that sum is same as ig
    output_attr[uncomputed_mask] = gain_fun(uncomputed_mask, attr)
    masks_trace = [v[0] for v in sorted(masks_trace, key=lambda x: -x[1])]
    if np.any(uncomputed_mask):
        masks_trace.append(uncomputed_mask)
    if integer_segments:
        attr_ranks = np.zeros(shape=attr.shape, dtype=np.int)
        for i, mask in enumerate(masks_trace):
            attr_ranks[mask] = i + 1
        return output_attr, attr_ranks
    else:
        return output_attr, masks_trace


class XRAI(SaliencyMask):
    def __init__(self, model):
        super(XRAI, self).__init__(model)
        self.integrated_gradients = IntegratedGradients(model)

    def get_mask(self, image_tensor, target_class=None, steps=100):
        bbl = self.integrated_gradients.get_mask(image_tensor, target_class, baseline='black', steps=steps)
        wbl = self.integrated_gradients.get_mask(image_tensor, target_class, baseline='white', steps=steps)
        mean_bl = np.mean([bbl, wbl], axis=0)
        baseline = np.max(mean_bl, axis=-1)

        image = np.moveaxis(image_tensor.detach().cpu().numpy()[0], 0, -1)
        segments = _get_segments_felzenszwalb(image)
        attr_map, _ = _xrai(baseline, segments)
        return attr_map
